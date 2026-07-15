import unittest

from core.device_scan import UsbPresentDevice, parse_pnp_device_lines


class DeviceScanTests(unittest.TestCase):
    def test_parse_pnp_device_lines_handles_multiple_devices(self) -> None:
        lines = [
            "Scarlett 2i2 USB|USB\\VID_1234&PID_5678",
            "Focusrite Scarlett Solo|USB\\VID_5678&PID_9012",
        ]

        devices = parse_pnp_device_lines(lines)

        self.assertEqual(2, len(devices))
        self.assertEqual("Scarlett 2i2 USB", devices[0].friendly_name)
        self.assertEqual("USB\\VID_1234&PID_5678", devices[0].instance_id)
        self.assertEqual("Focusrite Scarlett Solo", devices[1].friendly_name)

    def test_parse_pnp_device_lines_ignores_empty_entries(self) -> None:
        devices = parse_pnp_device_lines(["", "   ", "Nope"])
        self.assertEqual(0, len(devices))


if __name__ == "__main__":
    unittest.main()
