import unittest
from unittest.mock import patch

from app import resource_debug


class ResourceDebugTests(unittest.TestCase):
    def test_resource_delta_tracks_only_comparable_usage(self) -> None:
        start = {
            "ram": {"backend_rss_mb": 100.0, "system_used_mb": 2000.0},
            "vram": {"available": True, "used_mb": 4000},
        }
        end = {
            "ram": {"backend_rss_mb": 112.5, "system_used_mb": 2015.0},
            "vram": {"available": True, "used_mb": 4096},
        }

        self.assertEqual(
            resource_debug.resource_delta(start, end),
            {
                "ram.backend_rss_mb": 12.5,
                "ram.system_used_mb": 15.0,
                "vram.used_mb": 96,
            },
        )

    def test_gpu_memory_is_optional(self) -> None:
        with patch.object(resource_debug, "NVIDIA_SMI", None):
            self.assertEqual(resource_debug._gpu_memory(), {"available": False})


if __name__ == "__main__":
    unittest.main()
