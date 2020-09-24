
from skills.skill import Skill, RunConfig

class GeneralClassifierSkill(Skill):
    OOS_CUTOFF = 0.7
    infraset_model_name = 'infraset_model.torch'
    ffcc_keras_model_name = "ffcc_keras_model"
    label_encoding_name = 'label_encoding.pkl'
    def __init__(run_config, model_folder, model_download_paths=[]):
        if not run_config:
            self.run_config = RunConfig(local_server=True)
        super().__init__(run_config=run_config,
                        model_folder=model_folder, 
                        model_download_paths=model_files_download_paths) 
        self.infraset_model = None
        self.ffcc = None
        self.label_encoding = None
    
    def init_models():
        infraset_model_path = os.path.join(self.model_folder, self.infraset_model_name)
        self.infraset_model = torch.load(infraset_model_path)
        self.infraset_model.eval()

        keras_model_path = os.path.join(self.model_folder, self.ffcc_keras_model_name)
        self.ffcc = tf.keras.models.load_model(keras_model_path)

        label_encoding_path = os.path.join(self.model_folder, self.label_encoding_name)
        self.label_encoder = pickle.load(open(label_encoding_path, 'rb'))

    def get_labels_decoded(self, arr):
        return self.label_encoder.inverse_transform(arr)

    def get_doc2vec(self, text, verbose=False):
        return self.infraset_model.encode(text, verbose=verbose)

    def classify(self, text):
        vectorized_text = get_doc2vec([text])
        predictions = ffcc.predict(vectorized_text)
        predicted_labels = get_labels_decoded(np.argmax(predictions, axis=1))  
        label = predicted_labels[0]
        prob = np.max(predictions, axis=1) 

        if prob < self.OOS_CUTOFF:
            label = 'oos'  # since only one item

        return prob, cutoff
        