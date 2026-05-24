```python
from django.db import models
from django.test import TestCase
from django.db import connection
from django.db.models import QuerySet


class Item(models.Model):
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.name


def union_with_slicing_check(qs1, qs2):
    """
    Combines two querysets using union(), ensuring slicing is respected.
    This function wraps the union operation and checks if slicing is applied correctly.
    If the database backend ignores slicing during union, it converts the sliced querysets
    into subqueries to enforce the limits and offsets.
    """

    def enforce_slicing(qs):
        if qs.query.low_mark or qs.query.high_mark:
            return qs.from_db(qs._db, qs._queryset_class(qs.model, using=qs._db).filter(pk__in=qs.values_list('pk', flat=True)))
        return qs

    if connection.vendor == 'sqlite':
        qs1 = enforce_slicing(qs1)
        qs2 = enforce_slicing(qs2)
    
    return qs1.union(qs2)



class UnionWithSlicingTest(TestCase):
    def setUp(self):
        Item.objects.create(name="Item 1")
        Item.objects.create(name="Item 2")
        Item.objects.create(name="Item 3")
        Item.objects.create(name="Item 4")
        Item.objects.create(name="Item 5")

    def test_union_with_slicing(self):
        qs1 = Item.objects.all()[:2]
        qs2 = Item.objects.all()[3:]

        combined_qs = union_with_slicing_check(qs1, qs2)

        self.assertEqual(combined_qs.count(), 4)
        names = sorted([item.name for item in combined_qs])
        self.assertEqual(names, ['Item 1', 'Item 2', 'Item 4', 'Item 5'])

    def test_union_with_slicing_empty_queryset(self):
        qs1 = Item.objects.all()[:0]  # Empty queryset
        qs2 = Item.objects.all()[3:]

        combined_qs = union_with_slicing_check(qs1, qs2)

        self.assertEqual(combined_qs.count(), 2)
        names = sorted([item.name for item in combined_qs])
        self.assertEqual(names, ['Item 4', 'Item 5'])

    def test_union_with_slicing_no_slice(self):
        qs1 = Item.objects.filter(name__startswith="Item")  # No slice
        qs2 = Item.objects.all()[3:]

        combined_qs = union_with_slicing_check(qs1, qs2)
        expected_count = Item.objects.filter(name__startswith="Item").count() + Item.objects.all()[3:].count() - len(set(Item.objects.filter(name__startswith="Item").values_list('pk', flat=True)).intersection(set(Item.objects.all()[3:].values_list('pk', flat=True))))
        self.assertEqual(combined_qs.count(), expected_count)

    def test_union_with_slicing_overlap(self):
        qs1 = Item.objects.all()[:3]
        qs2 = Item.objects.all()[2:4]

        combined_qs = union_with_slicing_check(qs1, qs2)
        
        # Overlap on "Item 3".  Union should have it once only
        names = sorted([item.name for item in combined_qs])
        self.assertEqual(len(names), 4)
        self.assertEqual(names, ['Item 1', 'Item 2', 'Item 3', 'Item 4'])

    def test_union_with_slicing_complex(self):
        qs1 = Item.objects.filter(name__icontains="1")[:1]
        qs2 = Item.objects.filter(name__icontains="5")[0:1]
        qs3 = Item.objects.filter(name__icontains="2")[0:1]

        combined_qs = union_with_slicing_check(union_with_slicing_check(qs1, qs2), qs3)

        self.assertEqual(combined_qs.count(), 3)
        names = sorted([item.name for item in combined_qs])
        self.assertEqual(names, ['Item 1', 'Item 2', 'Item 5'])
```