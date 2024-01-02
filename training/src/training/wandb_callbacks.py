import wandb
from wandb import WandbEvalCallback
import tensorflow as tf
from typing import List, Optional


class WandbClfEvalCallback(WandbEvalCallback):
    def __init__(
        self, 
        validation_data, 
        data_table_columns, 
        pred_table_columns,
        CFG, 
        num_batches=40, 
        id2label=None,
    ):
        super().__init__(data_table_columns, pred_table_columns)
        self.validation_data = validation_data.take(num_batches)
        self.id2label = id2label
        self.CFG = CFG

    def add_ground_truth(self, logs=None):
        for b, (images, labels) in enumerate(self.validation_data):
            for idx, (image, label) in enumerate(zip(images, labels)):
                image, label = image.numpy(), label.numpy()
                if self.id2label:
                    species = self.id2label[label]
                    self.data_table.add_data((b*self.CFG.BATCH_SIZE + idx), wandb.Image(image), label, species)
                else:
                    self.data_table.add_data((b*self.CFG.BATCH_SIZE + idx), wandb.Image(image), label)

    def add_model_predictions(self, logs=None):
        for b, (images, labels) in enumerate(self.validation_data):
            prediction_batch = self.model.predict(images, verbose=0)
            for idx, prediction in enumerate(prediction_batch):
                data_table_ref = self.data_table_ref
                # table_idxs = data_table_ref.get_index()
                pred = tf.argmax(prediction, axis=-1)
                if self.id2label:
                    species = self.id2label[pred.numpy()]
                    self.pred_table.add_data(
                        data_table_ref.data[idx + (b * self.CFG.BATCH_SIZE)][0],
                        data_table_ref.data[idx + (b * self.CFG.BATCH_SIZE)][1],
                        data_table_ref.data[idx + (b * self.CFG.BATCH_SIZE)][2],
                        data_table_ref.data[idx + (b * self.CFG.BATCH_SIZE)][3],
                        pred,
                        species,
                    )
                else:
                    self.pred_table.add_data(
                        data_table_ref.data[idx + (b * self.CFG.BATCH_SIZE)][0],
                        data_table_ref.data[idx + (b * self.CFG.BATCH_SIZE)][1],
                        data_table_ref.data[idx + (b * self.CFG.BATCH_SIZE)][2],
                        pred,
                    )                    

    def log_pred_table(
        self,
        type: str = "evaluation",
        table_name: str = "eval_data",
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Log the W&B Tables for model evaluation.

        The table will be logged multiple times creating new version. Use this
        to compare models at different intervals interactively.

        Args:
            type: (str) The type of the artifact, which is used to organize and
                differentiate artifacts. (default is 'evaluation')
            table_name: (str) The name of the table as will be displayed in the UI.
                (default is 'eval_data')
            aliases: (List[str]) List of aliases for the prediction table.
        """
        assert wandb.run is not None
        wandb.run.log({"pred table": self.pred_table})
        pred_artifact = wandb.Artifact(f"run_{wandb.run.id}_pred", type=type)
        pred_artifact.add(self.pred_table, table_name)
        wandb.run.log_artifact(pred_artifact, aliases=aliases or ["latest"])

    def log_data_table(
        self, name: str = "val", type: str = "dataset", table_name: str = "val_data"
    ) -> None:
        """Log the `data_table` as W&B artifact and call `use_artifact` on it.

        This lets the evaluation table use the reference of already uploaded data
        (images, text, scalar, etc.) without re-uploading.

        Args:
            name: (str) A human-readable name for this artifact, which is how you can
                identify this artifact in the UI or reference it in use_artifact calls.
                (default is 'val')
            type: (str) The type of the artifact, which is used to organize and
                differentiate artifacts. (default is 'dataset')
            table_name: (str) The name of the table as will be displayed in the UI.
                (default is 'val_data').
        """
        wandb.run.log({table_name: self.data_table})
        data_artifact = wandb.Artifact(name, type=type)
        data_artifact.add(self.data_table, table_name)

        # Calling `use_artifact` uploads the data to W&B.
        assert wandb.run is not None
        wandb.run.use_artifact(data_artifact)
        data_artifact.wait()

        # We get the reference table.
        self.data_table_ref = data_artifact.get(table_name)

    def on_train_end(self, logs=None):
        self.init_pred_table(column_names=self.pred_table_columns)
        self.add_model_predictions(logs)
        self.log_pred_table()

    def on_epoch_end(self, epoch, logs=None):
        pass