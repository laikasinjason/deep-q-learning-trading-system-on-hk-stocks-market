from keras.models import model_from_yaml
import keras

def load_model(atari_model):
    # load YAML and create model
    yaml_file = open('atari_dqn_model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights("atari_dqn_model.h5")
    atari_model.model = loaded_model
    print("Loaded model from disk")
    
    loaded_model2 = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model2.load_weights("atari_dqn_model.h5")
    atari_model.target_model = loaded_model
    
    print("Loaded target model")
    # compile the loaded model before further use
    optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    atari_model.model.compile(optimizer, loss='mean_squared_error')
    atari_model.target_model.compile(optimizer, loss='mean_squared_error')
    
def save_model(agent):
    # serialize model to YAML
    model_yaml = agent.model.model.to_yaml()
    with open(str(agent.__class__.__name__)+".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights(str(agent.__class__.__name__)+".h5")
    print("Saved " + str(agent.__class__.__name__) + " model to disk")