import unittest
from unittest.mock import patch

from app import app
from src.ml_pipeline import get_prediction_fields


class GymChurnWebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config.update(TESTING=True)

    def setUp(self):
        self.client = app.test_client()

    def test_all_pages_render(self):
        for path in ["/", "/predict", "/dataset", "/evaluation", "/about"]:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)

    def test_dataset_report_downloads_csv(self):
        response = self.client.get("/dataset/report.csv")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response.content_type)
        self.assertIn(b"name,dtype,missing,unique,description", response.data)

    def test_dashboard_and_dataset_apis_return_data(self):
        dashboard = self.client.get("/api/dashboard")
        dataset = self.client.get("/api/dataset")
        self.assertEqual(dashboard.status_code, 200)
        self.assertEqual(dataset.status_code, 200)
        self.assertEqual(dashboard.get_json()["total_records"], 1200)
        self.assertEqual(dataset.get_json()["columns_count"], 12)

    def test_prediction_uses_expected_model_fields(self):
        payload = {}
        for field in get_prediction_fields():
            if field["kind"] == "number":
                payload[field["name"]] = field["default"]
            else:
                payload[field["name"]] = field["default"]

        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertIn(result["label"], ["Churn", "No Churn"])
        self.assertIn("probability", result)

    def test_prediction_rejects_missing_features(self):
        response = self.client.post("/api/predict", json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Missing fields", response.get_json()["message"])

    def test_prediction_rejects_invalid_category(self):
        payload = {
            field["name"]: field["default"] for field in get_prediction_fields()
        }
        category = next(
            field for field in get_prediction_fields() if field["kind"] == "category"
        )
        payload[category["name"]] = "not-a-valid-category"
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("valid option", response.get_json()["message"])

    def test_prediction_rejects_numeric_values_outside_observed_range(self):
        payload = {
            field["name"]: field["default"] for field in get_prediction_fields()
        }
        age = next(
            field for field in get_prediction_fields() if field["name"] == "age"
        )
        payload["age"] = age["maximum"] + 1
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("observed dataset range", response.get_json()["message"])

    def test_training_endpoint_can_be_disabled_for_deployment(self):
        with patch.dict("os.environ", {"ENABLE_MODEL_TRAINING": "false"}):
            response = self.client.post("/api/train")
        self.assertEqual(response.status_code, 403)
        self.assertIn("disabled", response.get_json()["message"])


if __name__ == "__main__":
    unittest.main()
