
def skill_server_creator(param):
    def actual_decorator(func):
         def wrapper(*args, **kwargs):
            from flask import Flask
            app = Flask(__name__)
            @app.route("/classify/<str:msg>", methods=["POST","Get"])
            def classify(msg):
                return func(msg)
            return app
    return actual_decorator


if __name__ == "__main__":
    def test(str):
        return {"status": "testMessage"}
    app = skill_server_creator(test)
    app.run()

    