import pickle

def predict_release_risk(release_properties, path_to_model = 'trained/model_lr.pkl'):
    """
    Пример того что нужно передавать в `release_properties`
    release_properties = [[
        5, #'Importance of Business Processes'
        4, #'Technical Complexity'
        4, #'Team Experience'
        2, #'Level of Integration with Other Systems'
        1, #'Reaction to Mistakes'
        2, #'Criticality of Streams'   
    ]]
    """
    with open(path_to_model, 'rb') as f:
        m = pickle.load(f)
        
    return m.predict(release_properties)
