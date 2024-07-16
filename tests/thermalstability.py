from saprot.utils.weights import PretrainedModel


def get_thermol_model():
    weight_worker=PretrainedModel(dir='/Users/yyy/.REvoDesign/weights/SaProt/',huggingface_id='SaProtHub', model_name='Model-Thermostability-650M',loader_type='native')
    weight_worker._fetch_model()


def main():

    get_thermol_model()


if __name__ == '__main__':
    main()