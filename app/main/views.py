import os

from flask import render_template, request, jsonify, current_app, send_from_directory, session

from app import db
from app.main import main
from app.models.record import Record, Tag
from .. import socketio
from flask_socketio import emit, join_room, leave_room
import numpy as np
import tensorflow as tf
import torch
import pickle
import spacy, re
from transformers import pipeline
import torch
import time

spacy_tagger = spacy.load('en_core_web_sm')

infraset_model = torch.load('infraset_model.torch')
infraset_model.eval()
ffcc = tf.keras.models.load_model('ffcc_keras_model')
label_encoder = pickle.load(open('label_encoding.pkl', 'rb'))
BOT_NAME = 'cain'
default_question_answering = pipeline('question-answering')


def get_labels_decoded(arr):
    return label_encoder.inverse_transform(arr)


def get_doc2vec(text, verbose=False):
    return infraset_model.encode(text, verbose=verbose)


def quick_predict_label(text, cutoff=0.0):
    X = get_doc2vec([text])
    predictions = ffcc.predict(X)
    predicted_labels = get_labels_decoded(np.argmax(predictions, axis=1))  # each integer is [0, 0, 0,1]
    prob = np.max(predictions, axis=1)  # each integer is [0, 0, 0,1]
    if prob < cutoff:
        predicted_labels[0] = 'oos'  # since only one item

    return predicted_labels[0], prob[0]


@main.route("/")
def index():
    return render_template('index.html')


@main.route("/classify/<string:text>", methods=["POST", "GET"])
def classify(text):
    intent, prob = quick_predict_label(text)
    prob = "{}".format(float(prob))
    print(type(prob))
    # record = Record(intent_classification=intent, log=text)
    # db.session.add(record)
    # db.session.commit()
    return jsonify({"intent": intent, "prob": prob})

@main.route("/classify_record/<string:text>", methods=["POST", "GET"])
def classify_url(text):
    """Sent by a client when the user entered a new message.
    The message is sent to all people in the room."""
    request.namespace = 'namespace'
    room = session.get('room', 'default room')
    msg = text
    start = time.time()

    emit('message', {'msg': session.get('name', 'test') + ':' + msg}, room=room)
    record = Record(log=msg)
    db.session.add(record)
    intent, prob = quick_predict_label(msg, cutoff=0.5)
    record.intent_classification = intent
    log = []
    emit('message', {'msg': '<Classified Last Message from Cain as {} with prob {}'.format(intent, prob)},
         room=room)
    log.append({'msg': '<Classified Last Message from Cain as {} with prob {}'.format(intent, prob)})
    tagged_sent = []
    
    if prob > 0.50:
        doc = spacy_tagger(msg)
        tags = [(w.text, w.tag_) for w in doc if w.tag_ in ['NNP', 'NN', 'NNS']]
        tagged_sent = ['({}, {}) '.format(w.text, w.tag_) for w in doc if w.tag_ in ['NNP', 'NN', 'NNS']]
        emit('message', {'msg': '<Relevant Tags found are {} >'.format(' '.join(tagged_sent))}, room=room)
        log.append({'msg': '<Relevant Tags found are {} >'.format(' '.join(tagged_sent))})
        relevant_record_tags = []
        for item in tags:
            tag = Tag(keyword=item[0], pp_speech=item[1])
            db.session.add(tag)
            relevant_record_tags.append(tag)
        record.tags = relevant_record_tags
        db.session.commit()

    db.session.commit()

    if BOT_NAME in msg:
        msg = msg.replace(BOT_NAME, '')

        context_records = Record.query.filter_by(intent_classification=intent).order_by(
            Record.created_date.desc()).limit(5)

        context = ' '.join([rec.log for rec in context_records])
        emit('message', {'msg': '<Context for message: {}>'.format(context)}, room=room)
        log.append({'msg': '<Context for message: {}>'.format(context)})
        resp = default_question_answering({'context': context, 'question': msg})
        end = time.time()
        emit('message', {'msg': '!Bot: {}!'.format(resp.get('answer'))},
             room=room)
        log.append({'msg': '!Bot: {}!'.format(resp.get('answer'))})
        emit('message',
             {'msg': '<Bot answered with score {} and with time {}s>'.format(resp.get('score'), -start + end)},
             room=room)
        log.append({'msg': '<Bot answered with score {} and with time {}s>'.format(resp.get('score'), -start + end)})

    return jsonify({
        'log': log
    })


@socketio.on('joined', namespace='/chat')
def joined(message):
    """Sent by clients when they enter a room.
    A status message is broadcast to all people in the room."""
    room = session.get('room', 'default room')
    join_room(room)
    emit('status', {'msg': session.get('name', 'test') + ' has entered the room.'}, room=room)


@socketio.on('text', namespace='/chat')
def text(message):
    """Sent by a client when the user entered a new message.
    The message is sent to all people in the room."""
    room = session.get('room', 'default room')
    msg = message['msg']
    start = time.time()

    emit('message', {'msg': session.get('name', 'test') + ':' + message['msg']}, room=room)
    record = Record(log=msg)
    db.session.add(record)
    intent, prob = quick_predict_label(msg, cutoff=0.5)
    record.intent_classification = intent
    emit('message', {'msg': '<Classified Last Message from Cain as {} with prob {}'.format(intent, prob)},
         room=room)
    if prob > 0.50:
        doc = spacy_tagger(msg)
        tags = [(w.text, w.tag_) for w in doc if w.tag_ in ['NNP', 'NN', 'NNS']]
        tagged_sent = ['({}, {}) '.format(w.text, w.tag_) for w in doc if w.tag_ in ['NNP', 'NN', 'NNS']]
        emit('message', {'msg': '<Relevant Tags found are {} >'.format(' '.join(tagged_sent))}, room=room)
        relevant_record_tags = []
        for item in tags:
            tag = Tag(keyword=item[0], pp_speech=item[1])
            db.session.add(tag)
            relevant_record_tags.append(tag)
        record.tags = relevant_record_tags
        db.session.commit()

    db.session.commit()

    if BOT_NAME in msg:
        msg = msg.replace(BOT_NAME, '')

        context_records = Record.query.filter_by(intent_classification=intent).order_by(
            Record.created_date.desc()).limit(5)

        context = ' '.join([rec.log for rec in context_records])
        emit('message', {'msg': '<Context for message: {}>'.format(context)}, room=room)
        resp = default_question_answering({'context': context, 'question': msg})
        end = time.time()
        emit('message', {'msg': '!Bot: {}!'.format(resp.get('answer'))},
             room=room)

        emit('message',
             {'msg': '<Bot answered with score {} and with time {}s>'.format(resp.get('score'), -start + end)},
             room=room)


@socketio.on('left', namespace='/chat')
def left(message):
    """Sent by clients when they leave a room.
    A status message is broadcast to all people in the room."""
    room = session.get('room')
    leave_room(room)
    emit('status', {'msg': session.get('name', 'test') + ' has left the room.'}, room=room)


@socketio.on('session_name', namespace='/chat')
def session_name(message):
    """Sent by clients when they leave a room.
    A status message is broadcast to all people in the room."""
    room = session.get('room')
    session['name'] = message['msg']
    emit('status', {'msg': session.get('name', 'test') + ' has joined the room.'}, room=room)
