import pytest
from marshmallow import Schema, fields, ValidationError

class NameSchema(Schema):
    first = fields.Str()
    last = fields.Str()

class EmployeeSchema(Schema):
    name = fields.Nested(NameSchema)
    title = fields.Str()

class CompanySchema(Schema):
    employees = fields.Nested(EmployeeSchema, many=True)

def test_nested_many_with_null():
    """Test that nested schema with many=True raises TypeError when encountering null values."""
    company_data = {
        "employees": [
            {"name": {"first": "John", "last": "Doe"}, "title": "CEO"},
            None,  # Introducing a null value in the list
            {"name": {"first": "Jane", "last": "Smith"}, "title": "CTO"},
        ]
    }

    schema = CompanySchema()

    with pytest.raises(TypeError) as excinfo:
        schema.dump(company_data)

    assert "Object of type NoneType is not JSON serializable" in str(excinfo.value)


def test_nested_many_without_null():
    """Test that nested schema with many=True works correctly without null values."""
    company_data = {
        "employees": [
            {"name": {"first": "John", "last": "Doe"}, "title": "CEO"},
            {"name": {"first": "Jane", "last": "Smith"}, "title": "CTO"},
        ]
    }

    schema = CompanySchema()
    result = schema.dump(company_data)

    expected_result = {
        "employees": [
            {"name": {"first": "John", "last": "Doe"}, "title": "CEO"},
            {"name": {"first": "Jane", "last": "Smith"}, "title": "CTO"},
        ]
    }
    assert result == expected_result

def test_nested_many_allowing_none():
    """Test that nested schema with many=True works with None values when specified"""

    class CompanySchemaAllowNone(Schema):
        employees = fields.Nested(EmployeeSchema, many=True, allow_none=True)

    company_data = {
        "employees": [
            {"name": {"first": "John", "last": "Doe"}, "title": "CEO"},
            None,  # Introducing a null value in the list
            {"name": {"first": "Jane", "last": "Smith"}, "title": "CTO"},
        ]
    }

    schema = CompanySchemaAllowNone()
    
    with pytest.raises(ValidationError) as excinfo:
        schema.dump(company_data)
    
    assert "Invalid type." in str(excinfo.value)

def test_nested_many_nullable_true():
    """Test that nested schema with many=True works with None values when specified"""

    class CompanySchemaNullable(Schema):
        employees = fields.Nested(EmployeeSchema, many=True, nullable=True)

    company_data = {
        "employees": [
            {"name": {"first": "John", "last": "Doe"}, "title": "CEO"},
            None,  # Introducing a null value in the list
            {"name": {"first": "Jane", "last": "Smith"}, "title": "CTO"},
        ]
    }

    schema = CompanySchemaNullable()
    
    with pytest.raises(TypeError) as excinfo:
        schema.dump(company_data)
    
    assert "Object of type NoneType is not JSON serializable" in str(excinfo.value)

if __name__ == "__main__":
    pytest.main([__file__])