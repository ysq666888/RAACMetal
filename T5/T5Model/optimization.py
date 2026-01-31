from tensorflow.keras.optimizers import Adam


def create_optimizer(config):
    
    optimizer = Adam(
        learning_rate=config.initial_learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=1.5,           
        amsgrad=True        
    )
    
    return optimizer