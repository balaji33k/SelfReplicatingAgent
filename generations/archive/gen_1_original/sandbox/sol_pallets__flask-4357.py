from flask import Flask, Blueprint, url_for

app = Flask(__name__)

# Create blueprints
parent_bp = Blueprint('parent', __name__, url_prefix='/parent')
child_bp = Blueprint('child', __name__, url_prefix='/child')


# Define a view function within the child blueprint
@child_bp.route('/leaf')
def leaf_endpoint():
    return "This is the leaf endpoint."


# Register the child blueprint with the parent blueprint
parent_bp.register_blueprint(child_bp)

# Register the parent blueprint with the app
app.register_blueprint(parent_bp)


@app.route('/')
def index():
    # Test url_for from the app context
    leaf_url = url_for('parent.child.leaf_endpoint')
    return f"URL for leaf endpoint: {leaf_url}"


# Test case function
def test_url_for():
    with app.test_request_context():
        leaf_url = url_for('parent.child.leaf_endpoint')
        assert leaf_url == '/parent/child/leaf', f"Expected '/parent/child/leaf', but got {leaf_url}"
        print("Test case passed: URL for leaf endpoint is correctly generated.")


if __name__ == '__main__':
    # Run the test case
    test_url_for()

    # You can also run the Flask app to test in a browser
    # app.run(debug=True)