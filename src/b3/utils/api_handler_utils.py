import base64
import io
import joblib


class ApiHandlerUtils:
    @staticmethod
    def serialize_model(model):
        """Serialize model to base64 string for storage."""
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        model_bytes = buffer.getvalue()
        return base64.b64encode(model_bytes).decode('utf-8')

    @staticmethod
    def deserialize_model(model_b64):
        """Deserialize model from base64 string."""
        model_bytes = base64.b64decode(model_b64)
        buffer = io.BytesIO(model_bytes)
        return joblib.load(buffer)

    @staticmethod
    def make_log_api_activity(request_data_loader, executor):
        def _log_api_activity(endpoint, request_data, response_data, status, error_message=None):
            def log_task():
                if request_data_loader:
                    log_kwargs = {
                        'endpoint': endpoint,
                        'request_data': request_data,
                        'response_data': response_data,
                        'status': status
                    }
                    if error_message is not None:
                        log_kwargs['error_message'] = error_message
                    request_data_loader.log_api_activity(**log_kwargs)

            executor.submit(log_task)

        return _log_api_activity
