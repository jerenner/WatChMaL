"""
Class for training a fully supervised classifier
"""

# torch imports
import torch

# generic imports
import numpy as np
from time import strftime, localtime, time

# WatChMaL imports
from watchmal.engine.engine_base import BaseEngine

class ClassifierEngine:
    def __init__(self, model, rank, gpu, dump_path, label_set):
        """
        Args:
            model       ... model object that engine will use in training or evaluation
            rank        ... rank of process among all spawned processes (in multiprocessing mode)
            gpu         ... gpu that this process is running on
            dump_path   ... path to store outputs in
        """
        # create the directory for saving the log and dump files
        super().__init__(model, rank, gpu, dump_path)

        # Setup the parameters to save given the model type
        if isinstance(self.model, DDP):
            self.is_distributed = True
            self.model_accs = self.model.module
            self.ngpus = torch.distributed.get_world_size()
        else:
            self.is_distributed = False
            self.model_accs = self.model

        self.data_loaders = {}
        self.label_set = label_set

        # define the placeholder attributes
        self.data = None
        self.labels = None
        self.loss = None

        # logging attributes
        self.train_log = CSVData(self.dirpath + "log_train_{}.csv".format(self.rank))

        if self.rank == 0:
            self.val_log = CSVData(self.dirpath + "log_val.csv")

        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        
        self.optimizer = None
        self.scheduler = None
    
    def configure_optimizers(self, optimizer_config):
        """
        Set up optimizers from optimizer config

        Args:
            optimizer_config    ... hydra config specifying optimizer object
        """
        self.optimizer = instantiate(optimizer_config, params=self.model_accs.parameters())

  
    def configure_scheduler(self, scheduler_config):
        """
        Set up scheduler from scheduler config

        Args:
            scheduler_config    ... hydra config specifying scheduler object
        """
        self.scheduler = instantiate(scheduler_config, optimizer=self.optimizer)
        print('Successfully set up Scheduler')


    def configure_data_loaders(self, data_config, loaders_config, is_distributed, seed):
        """
        Set up data loaders from loaders config

        Args:
            data_config     ... hydra config specifying dataset
            loaders_config  ... hydra config specifying dataloaders
            is_distributed  ... boolean indicating if running in multiprocessing mode
            seed            ... seed to use to initialize dataloaders
        
        Parameters:
            self should have dict attribute data_loaders
        """
        for name, loader_config in loaders_config.items():
            self.data_loaders[name] = get_data_loader(**data_config, **loader_config, is_distributed=is_distributed, seed=seed)
            if self.label_set is not None:
                self.data_loaders[name].map_labels(self.label_set)
    
    def get_synchronized_metrics(self, metric_dict):
        """
        Gathers metrics from multiple processes using pytorch distributed operations

        Args:
            metric_dict         ... dict containing values that are tensor outputs of a single process
        
        Returns:
            global_metric_dict  ... dict containing concatenated list of tensor values gathered from all processes
        """
        global_metric_dict = {}
        for name, array in zip(metric_dict.keys(), metric_dict.values()):
            tensor = torch.as_tensor(array).to(self.device)
            global_tensor = [torch.zeros_like(tensor).to(self.device) for i in range(self.ngpus)]
            torch.distributed.all_gather(global_tensor, tensor)
            global_metric_dict[name] = torch.cat(global_tensor)
        
        return global_metric_dict

    def forward(self, train=True):
        """
        Compute predictions and metrics for a batch of data

        Args:
            train   ... whether to compute gradients for backpropagation

        Parameters:
            self should have attributes data, labels, model, criterion, softmax

        Returns:
            dict containing loss, predicted labels, softmax, accuracy, and raw model outputs
        """

        with torch.set_grad_enabled(train):
            # Move the data and the labels to the GPU (if using CPU this has no effect)
            data = self.data.to(self.device)
            labels = self.labels.to(self.device)
            model_out = self.model(data)
            softmax = self.softmax(model_out)
            predicted_labels = torch.argmax(model_out, dim=-1)
            self.loss = self.criterion(model_out, labels)
            accuracy = (predicted_labels == labels).sum().item() / float(predicted_labels.nelement())
            result = {'predicted_labels': predicted_labels,
                      'softmax': softmax,
                      'raw_pred_labels': model_out,
                      'loss': self.loss.item(),
                      'accuracy': accuracy}
        return result

    # ========================================================================
    # Training and evaluation loops

    def train(self, train_config):
        """
        Train the model on the training set

        Args:
            train_config    ... config specigying training parameters

        Parameters:
            self should have attributes model, data_loaders

        Outputs:
            val_log      ... csv log containing iteration, epoch, loss, accuracy for each iteration on validation set
            train_logs   ... csv logs containing iteration, epoch, loss, accuracy for each iteration on training set

        Returns: None
        """
        # initialize training params
        epochs = train_config.epochs
        report_interval = train_config.report_interval
        val_interval = train_config.val_interval
        num_val_batches = train_config.num_val_batches
        checkpointing = train_config.checkpointing
        save_interval = train_config.save_interval if 'save_interval' in train_config else None

        # set the iterations at which to dump the events and their metrics
        if self.rank == 0:
            print(f"Training... Validation Interval: {val_interval}")

        # set model to training mode
        self.model.train()

        # initialize epoch and iteration counters
        self.epoch = 0
        self.iteration = 0
        self.step = 0
        # keep track of the validation loss
        self.best_validation_loss = 1.0e10

        # initialize the iterator over the validation set
        val_iter = iter(self.data_loaders["validation"])

        # global training loop for multiple epochs
        for self.epoch in range(epochs):
            if self.rank == 0:
                print('Epoch', self.epoch+1, 'Starting @', strftime("%Y-%m-%d %H:%M:%S", localtime()))

            start_time = time()
            iteration_time = start_time

            train_loader = self.data_loaders["train"]
            self.step = 0
            # update seeding for distributed samplers
            if self.is_distributed:
                train_loader.sampler.set_epoch(self.epoch)

            # local training loop for batches in a single epoch
            steps_per_epoch = len(train_loader)
            for self.step, train_data in enumerate(train_loader):

                # run validation on given intervals
                if self.iteration % val_interval == 0:
                    self.validate(val_iter, num_val_batches, checkpointing)

                # Train on batch
                self.data = train_data['data']
                self.labels = train_data['labels']

                # Call forward: make a prediction & measure the average error using data = self.data
                res = self.forward(True)

                # Call backward: backpropagate error and update weights using loss = self.loss
                self.backward()

                # update the epoch and iteration
                # self.epoch += 1. / len(self.data_loaders["train"])
                self.step += 1
                self.iteration += 1

                # get relevant attributes of result for logging
                train_metrics = {"iteration": self.iteration, "epoch": self.epoch, "loss": res["loss"], "accuracy": res["accuracy"]}
                
                # record the metrics for the mini-batch in the log
                self.train_log.record(train_metrics)
                self.train_log.write()
                self.train_log.flush()

                # print the metrics at given intervals
                if self.rank == 0 and self.iteration % report_interval == 0:
                    previous_iteration_time = iteration_time
                    iteration_time = time()
                    print("... Iteration %d ... Epoch %d ... Step %d/%d  ... Training Loss %1.3f ... Training Accuracy %1.3f ... Time Elapsed %1.3f ... Iteration Time %1.3f" %
                          (self.iteration, self.epoch + 1, self.step, steps_per_epoch, res["loss"], res["accuracy"], iteration_time - start_time, iteration_time - previous_iteration_time))

            if self.scheduler is not None:
                self.scheduler.step()

            if self.rank == 0 and (save_interval is not None) and ((self.epoch+1)%save_interval == 0):
                self.save_state(name=f'_epoch_{self.epoch+1}')

        self.train_log.close()
        if self.rank == 0:
            self.val_log.close()

    def validate(self, val_iter, num_val_batches, checkpointing):
        # set model to eval mode
        self.model.eval()
        val_metrics = {"iteration": self.iteration, "loss": 0., "accuracy": 0., "saved_best": 0}
        for val_batch in range(num_val_batches):
            try:
                val_data = next(val_iter)
            except StopIteration:
                del val_iter
                print("Fetching new validation iterator...")
                val_iter = iter(self.data_loaders["validation"])
                val_data = next(val_iter)

            # extract the event data from the input data tuple
            self.data = val_data['data']
            self.labels = val_data['labels']

            val_res = self.forward(False)

            val_metrics["loss"] += val_res["loss"]
            val_metrics["accuracy"] += val_res["accuracy"]
        # return model to training mode
        self.model.train()

        # record the validation stats to the csv
        val_metrics["loss"] /= num_val_batches
        val_metrics["accuracy"] /= num_val_batches
        local_val_metrics = {"loss": np.array([val_metrics["loss"]]), "accuracy": np.array([val_metrics["accuracy"]])}

        if self.is_distributed:
            global_val_metrics = self.get_synchronized_metrics(local_val_metrics)
            for name, tensor in zip(global_val_metrics.keys(), global_val_metrics.values()):
                global_val_metrics[name] = np.array(tensor.cpu())
        else:
            global_val_metrics = local_val_metrics

        if self.rank == 0:
            # Save if this is the best model so far
            global_val_loss = np.mean(global_val_metrics["loss"])
            global_val_accuracy = np.mean(global_val_metrics["accuracy"])

            val_metrics["loss"] = global_val_loss
            val_metrics["accuracy"] = global_val_accuracy
            val_metrics["epoch"] = self.epoch

            if val_metrics["loss"] < self.best_validation_loss:
                self.best_validation_loss = val_metrics["loss"]
                print('best validation loss so far!: {}'.format(self.best_validation_loss))
                self.save_state("BEST")
                val_metrics["saved_best"] = 1

            # Save the latest model if checkpointing
            if checkpointing:
                self.save_state()

            self.val_log.record(val_metrics)
            self.val_log.write()
            self.val_log.flush()

    def evaluate(self, test_config):
        """
        Evaluate the performance of the trained model on the test set

        Args:
            test_config ... hydra config specifying evaluation parameters

        Parameters:
            self should have attributes model, data_loaders, dirpath

        Outputs:
            indices     ... index in dataset of each event
            labels      ... actual label of each event
            predictions ... predicted label of each event
            softmax     ... softmax output over classes for each event

        Returns: None
        """
        print("evaluating in directory: ", self.dirpath)

        # Variables to output at the end
        eval_loss = 0.0
        eval_acc = 0.0
        eval_iterations = 0

        # Iterate over the validation set to calculate val_loss and val_acc
        with torch.no_grad():

            # Set the model to evaluation mode
            self.model.eval()

            # Variables for the outputs
            # TODO: find some way of determining the softmax_shape without having to do a forward run
            self.data = next(iter(self.data_loaders["test"]))['data']
            self.labels = next(iter(self.data_loaders["test"].dataset))['labels']
            softmax_shape = self.forward(train=False)['softmax'][0].shape
            indices = np.zeros((0,))
            labels = np.zeros((0,))
            predictions = np.zeros((0,))
            softmaxes = np.zeros((0, *softmax_shape))

            start_time = time()
            iteration_time = start_time

            # Extract the event data and label from the DataLoader iterator
            steps_per_epoch = len(self.data_loaders["test"])
            for it, eval_data in enumerate(self.data_loaders["test"]):

                # load data
                self.data = eval_data['data']
                self.labels = eval_data['labels']

                eval_indices = eval_data['indices']

                # Run the forward procedure and output the result
                result = self.forward(train=False)

                eval_loss += result['loss']
                eval_acc  += result['accuracy']

                # Add the local result to the final result
                indices = np.concatenate((indices, eval_indices))
                labels = np.concatenate((labels, self.labels))
                predictions = np.concatenate((predictions, result['predicted_labels'].detach().cpu().numpy()))
                softmaxes = np.concatenate((softmaxes, result['softmax'].detach().cpu().numpy()))

                eval_iterations += 1

                # print the metrics at given intervals
                if self.rank == 0 and it % test_config.report_interval == 0:
                    previous_iteration_time = iteration_time
                    iteration_time = time()
                    print("... Iteration %d / %d ... Evaluation Loss %1.3f ... Evaluation Accuracy %1.3f ... Time Elapsed %1.3f ... Iteration Time %1.3f" %
                          (it, steps_per_epoch, result['loss'], result['accuracy'], iteration_time - start_time, iteration_time - previous_iteration_time))

        print("loss : " + str(eval_loss / eval_iterations) + " accuracy : " + str(eval_acc/eval_iterations))

        iterations = np.array([eval_iterations])
        loss = np.array([eval_loss])
        accuracy = np.array([eval_acc])

        local_eval_metrics_dict = {"eval_iterations": iterations, "eval_loss": loss, "eval_acc": accuracy}

        indices     = np.array(indices)
        labels      = np.array(labels)
        predictions = np.array(predictions)
        softmaxes   = np.array(softmaxes)

        local_eval_results_dict = {"indices": indices, "labels": labels, "predictions": predictions, "softmaxes": softmaxes}

        if self.is_distributed:
            # Gather results from all processes
            global_eval_metrics_dict = self.get_synchronized_metrics(local_eval_metrics_dict)
            global_eval_results_dict = self.get_synchronized_metrics(local_eval_results_dict)

            if self.rank == 0:
                for name, tensor in zip(global_eval_metrics_dict.keys(), global_eval_metrics_dict.values()):
                    local_eval_metrics_dict[name] = np.array(tensor.cpu())

                indices     = global_eval_results_dict["indices"].cpu()
                labels      = global_eval_results_dict["labels"].cpu()
                predictions = global_eval_results_dict["predictions"].cpu()
                softmaxes   = global_eval_results_dict["softmaxes"].cpu()

        if self.rank == 0:
            # Save overall evaluation results
            print("Saving Data...")
            np.save(self.dirpath + "indices.npy", indices)
            np.save(self.dirpath + "labels.npy", labels)
            np.save(self.dirpath + "predictions.npy", predictions)
            np.save(self.dirpath + "softmax.npy", softmaxes)

            # Compute overall evaluation metrics
            val_iterations = np.sum(local_eval_metrics_dict["eval_iterations"])
            val_loss = np.sum(local_eval_metrics_dict["eval_loss"])
            val_acc = np.sum(local_eval_metrics_dict["eval_acc"])

            print("\nAvg eval loss : " + str(val_loss / val_iterations),
                  "\nAvg eval acc : "  + str(val_acc / val_iterations))
