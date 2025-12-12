from b3.service.web_api.b3_model_api import B3ModelAPI
from constants import SECONDARY_API_PORT
import logging

logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    B3ModelAPI().run(port=SECONDARY_API_PORT)
