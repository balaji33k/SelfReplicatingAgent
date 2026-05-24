import unittest


class TestPylintTypeInference(unittest.TestCase):
    def test_dynamic_attribute_in_try_except(self):
        class MyClass:
            def __init__(self):
                pass

        obj = MyClass()

        try:
            obj.dynamic_attribute = 10  # Dynamically add attribute
        except Exception:
            pass

        # Pylint might incorrectly flag this as 'no-member'
        # This test aims to demonstrate that the attribute *is* present
        # even if pylint's inference is wrong.

        self.assertTrue(hasattr(obj, 'dynamic_attribute'))
        self.assertEqual(obj.dynamic_attribute, 10)  # Verify the value

    def test_dynamic_attribute_with_else(self):
        class MyClass:
            def __init__(self):
                pass

        obj = MyClass()

        try:
            # Simulate a condition that *succeeds*
            result = 1
        except Exception:
            pass
        else:
            obj.dynamic_attribute = result * 10

        self.assertTrue(hasattr(obj, 'dynamic_attribute'))
        self.assertEqual(obj.dynamic_attribute, 10)

    def test_dynamic_attribute_in_except_block(self):
        class MyClass:
            def __init__(self):
                pass

        obj = MyClass()

        try:
            raise ValueError("Simulated error") # force except clause
        except ValueError:
            obj.dynamic_attribute = 20  # Dynamically add attribute

        self.assertTrue(hasattr(obj, 'dynamic_attribute'))
        self.assertEqual(obj.dynamic_attribute, 20)

    def test_no_exception(self):
        class MyClass:
            def __init__(self):
                pass

        obj = MyClass()
        try:
            obj.dynamic_attribute = 30
        except Exception as e:
            self.fail(f"Exception raised: {e}")

        self.assertTrue(hasattr(obj, 'dynamic_attribute'))
        self.assertEqual(obj.dynamic_attribute, 30)

    def test_nested_try_except(self):
        class MyClass:
            def __init__(self):
                pass

        obj = MyClass()

        try:
            try:
                obj.dynamic_attribute = 40
            except Exception:
                pass
        except Exception:
            pass

        self.assertTrue(hasattr(obj, 'dynamic_attribute'))
        self.assertEqual(obj.dynamic_attribute, 40)


if __name__ == '__main__':
    unittest.main()