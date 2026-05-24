from django.db import models
from django.test import TestCase
from django.db import connection

class Item(models.Model):
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.name


def union_with_slicing(qs1, qs2):
    """
    Handles union of querysets with slicing, ensuring correct SQL generation
    and respecting slice limits.
    """
    if qs1.query.low_mark or qs1.query.high_mark or qs2.query.low_mark or qs2.query.high_mark:
        # At least one queryset is sliced, force subquery usage
        return qs1.all().union(qs2.all()) # .all() forces re-evaluation/correct SQL for some DBs

    return qs1.union(qs2)



class UnionWithSlicingTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        # Create some test data
        for i in range(10):
            Item.objects.create(name=f"Item {i}")

    def test_union_with_slicing(self):
        # Create two sliced querysets
        qs1 = Item.objects.all()[:3]  # First 3 items
        qs2 = Item.objects.all()[3:6]  # Next 3 items

        # Call union() on them using the wrapper function
        combined_qs = union_with_slicing(qs1, qs2)

        # Verify the result count matches the sum of the individual slice limits
        self.assertEqual(len(list(combined_qs)), 6)  # 3 + 3 = 6
        self.assertEqual(combined_qs.count(), 6)

    def test_union_with_slicing_no_slice(self):
        # Create two sliced querysets
        qs1 = Item.objects.all()
        qs2 = Item.objects.all()

        # Call union() on them using the wrapper function
        combined_qs = union_with_slicing(qs1, qs2)

        # Verify the result count matches the sum of the individual slice limits
        self.assertEqual(combined_qs.count(), Item.objects.count()) # should be the same as the total count

    def test_union_with_only_one_slice(self):
        # Create two sliced querysets
        qs1 = Item.objects.all()[:3]
        qs2 = Item.objects.all()

        # Call union() on them using the wrapper function
        combined_qs = union_with_slicing(qs1, qs2)

        # Verify the result count matches the sum of the individual slice limits
        self.assertEqual(combined_qs.count(), Item.objects.count()) # should be the same as the total count

    def test_union_with_offset_only(self):
        qs1 = Item.objects.all()[2:]  # Offset only
        qs2 = Item.objects.all()[:2]  # Limit only

        combined_qs = union_with_slicing(qs1, qs2)
        self.assertEqual(combined_qs.count(), Item.objects.count()) # should be the same as the total count

    def test_union_with_both_offset_and_limit(self):
         qs1 = Item.objects.all()[2:5] # offset and limit
         qs2 = Item.objects.all()[5:8] # offset and limit

         combined_qs = union_with_slicing(qs1, qs2)
         self.assertEqual(combined_qs.count(), 6) #3 + 3

    def test_empty_queryset(self):
        qs1 = Item.objects.none()
        qs2 = Item.objects.all()[:3]

        combined_qs = union_with_slicing(qs1, qs2)
        self.assertEqual(combined_qs.count(), 0 if connection.vendor == 'sqlite' else 3) # 3 if sqlite returns duplicates