import boto3
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
from io import BytesIO
import random
import json
from spacy_model_manager import get_model



def s3_model_build_save():

    # Initialize the S3 client
    s3_client = boto3.client('s3')

    # Specify the S3 bucket name and PDF file key (path)
    bucket_name_ner = 'gbidl-us-west-2-test-realestate'
    json_file_key = 'gbi_re_dw/landing/Montague_new.json'

    try:

        # Step 1: Load the existing small English model
        #nlp = spacy.load("en_core_web_sm")
        nlp = get_model("en_core_web_sm")

        # Step 2: Create a new blank NLP pipeline
        nlp_new = spacy.blank("en")

        # Step 3: Add the custom NER component to the new pipeline
        ner_new = nlp_new.add_pipe("ner", source=nlp)

        # Step 4: Load the training data from JSON in S3
        TRAINING_DATA = []
        response_json = s3_client.get_object(Bucket=bucket_name_ner, Key=json_file_key)
        json_data = response_json['Body'].read()

        for data in json.loads(json_data):
            text = data["text"]
            entities = data["label"]
            entity_tuples = []
            for entity in entities:
                start = entity["start"]
                end = entity["end"]
                label = str(entity["labels"][0])  # Convert label to string format
                entity_tuples.append((start, end, label))
            TRAINING_DATA.append((text, {"entities": entity_tuples}))

        # Step 5: Add labels to the custom NER component
        for _, annotations in TRAINING_DATA:
            for ent in annotations.get("entities"):
                ner_new.add_label(ent[2])

        # Step 6: Training the custom NER model
        n_iter = 70
        for itn in range(n_iter):
            losses = {}
            random.shuffle(TRAINING_DATA)
            batches = minibatch(TRAINING_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                examples = []
                for text, annotation in zip(texts, annotations):
                    doc = nlp_new.make_doc(text)
                    example = Example.from_dict(doc, annotation)
                    examples.append(example)

                nlp_new.update(examples, losses=losses)

            print(f"Iteration {itn+1}/{n_iter}, Losses: {losses}")

        # Step 7: Merge the custom NER component with the existing small English model
        for ent in ner_new.move_names:
            ner = nlp.get_pipe("ner")
            ner.add_label(ent)

        # Save the entire trained model to S3
        output_model_key = 'gbi_re_dw/landing/custom_ner_model'
        output_stream = BytesIO()
        nlp.to_disk(output_stream)
        output_stream.seek(0)
        s3_client.upload_fileobj(output_stream, bucket_name_ner, output_model_key)

        s3_client.put_object_acl(Bucket=bucket_name_ner, Key=output_model_key, ACL='bucket-owner-full-control')

        print("Custom NER model training completed.")

    except Exception as e:
        print(f"An error occurred during NER training: {e}")
