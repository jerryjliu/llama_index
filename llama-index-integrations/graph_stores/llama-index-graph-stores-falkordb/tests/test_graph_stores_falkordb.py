import time
import docker
import unittest
from llama_index.graph_stores.falkordb.base import FalkorDBGraphStore

# Set up Docker client
docker_client = docker.from_env()


class TestFalkorDBGraphStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup method called once for the entire test class."""
        # Attempt to stop and remove the container if it already exists
        try:
            existing_container = docker_client.containers.get("falkordb_test_instance")
            existing_container.stop()
            existing_container.remove()
        except docker.errors.NotFound:
            pass  # If no container exists, we can proceed
        except Exception as e:
            print(f"Error while stopping/removing existing container: {e}")

        # Start FalkorDB container
        try:
            cls.container = docker_client.containers.run(
                "falkordb/falkordb:latest",
                detach=True,
                name="falkordb_test_instance",
                ports={"6379/tcp": 6379},
            )
            time.sleep(2)  # Allow time for the container to initialize
        except Exception as e:
            print(f"Error starting FalkorDB container: {e}")
            raise

        # Set up the FalkorDB store and clear database
        cls.graph_store = FalkorDBGraphStore(url="redis://localhost:6379")
        cls.graph_store.structured_query(
            "MATCH (n) DETACH DELETE n"
        )  # Clear the database

    @classmethod
    def tearDownClass(cls):
        """Teardown method called once after all tests are done."""
        try:
            cls.container.stop()
            cls.container.remove()
        except Exception as e:
            print(f"Error stopping/removing container: {e}")

    def test_upsert_triplet(self):
        # Call the method you want to test
        self.graph_store.upsert_triplet("node1", "related_to", "node2")

        # Check if the data has been inserted correctly
        result = self.graph_store.get("node1")  # Adjust the method to retrieve data
        expected_result = [
            "RELATED_TO",
            "node2",
        ]  # Adjust this based on what you expect
        self.assertIn(expected_result, result)

        result = self.graph_store.get_rel_map(["node1"], 1)
        self.assertIn(expected_result, result["node1"])

        self.graph_store.delete("node1", "related_to", "node2")

        result = self.graph_store.get("node1")  # Adjust the method to retrieve data
        expected_result = []  # Adjust this based on what you expect
        self.assertEqual(expected_result, result)

        self.graph_store.switch_graph("new_graph")
        self.graph_store.refresh_schema()


if __name__ == "__main__":
    unittest.main()
