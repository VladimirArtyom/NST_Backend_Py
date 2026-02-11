import train as t
import utils as u
from models.definitions import vgg_nets
if __name__ == "__main__":
    t_config = t.TrainingConfiguration()
    
    train_dataLoader, test_dataLoader, cl_namae = t.get_dataloaders(t_config)
    model = t.create_model_for_training(cl_namae)
    trained_model, history =t.train_model(model, train_dataLoader, test_dataLoader, t_config, cl_namae)
    u.plot_training_history(history)