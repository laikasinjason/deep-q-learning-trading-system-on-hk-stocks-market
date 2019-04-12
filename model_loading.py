import keras
from keras.models import model_from_yaml

from sell_signal_agent import SellSignalAgent


def load_model(agent, loss_function='mean_squared_error'):
    # load YAML and create model
    yaml_file = open(str(agent.__class__.__name__) + '.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights(str(agent.__class__.__name__) + ".h5")
    agent.model.model = loaded_model
    print(str(agent.__class__.__name__) + " - Loaded model from disk")

    # compile the loaded model before further use
    optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    agent.model.model.compile(optimizer, loss=loss_function)

    if isinstance(agent, SellSignalAgent):
        loaded_model2 = model_from_yaml(loaded_model_yaml)
        agent.model.target_model = loaded_model2
        agent.model.target_model.compile(optimizer, loss=loss_function)


def save_model(agent):
    # serialize model to YAML
    model_yaml = agent.model.model.to_yaml()
    with open(str(agent.__class__.__name__) + ".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    agent.model.model.save_weights(str(agent.__class__.__name__) + ".h5")
    print("Saved " + str(agent.__class__.__name__) + " model to disk")


def save_tf_model(agent):
    save_path = agent.model.saver.save(agent.model.sess, "./models/" + agent.__class__.__name__ + ".ckpt")
    print("Model Saved")
