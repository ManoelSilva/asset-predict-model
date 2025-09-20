from b3.service.web_api.b3_model_api import B3TrainingAPI

if __name__ == '__main__':
    B3TrainingAPI().run(port=5001, debug=True)
