from flask import Flask, request, jsonify
import os

class API: 
    def __init__(self,_QA):
        self.app = Flask(__name__)
        self.QA = _QA
        @self.app.route("/api/prompt", methods=["GET", "POST"])
        def prompt():
            prompt_user = request.form.get("prompt_user")
            if prompt_user == None:
                prompt_user = request.args.get("prompt_user")
            if prompt_user:
                response = self.QA(prompt_user)
                answer, docs = response["result"], response["source_documents"]

                dictionary_prompt_response = {
                    "Prompt": prompt_user,
                    "Answer": answer,
                }

                dictionary_prompt_response["Sources"] = []
                for document in docs:
                    dictionary_prompt_response["Sources"].append(
                        (os.path.basename(str(document.metadata["source"])), str(document.page_content))
                    )

                return jsonify(dictionary_prompt_response), 200
            return "No user prompt received", 400

    def ServerStart(self):
        self.app.run(debug=False, port=5110) # Run the Flask app when the run method is called
        print("Server started")

if __name__ == "__main__":
    print("Use Main.py instead (python3 main.py/python Main.py)")

