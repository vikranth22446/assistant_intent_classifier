import os

from flask import render_template, request, jsonify, current_app, send_from_directory, session

from app import db
from app.main import main
from app.models.record import Record, Tag, UserSkills
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
from app import global_skills_dic

import logging

log = logging.getLogger(__name__)

# from app.voice.shopping_main import find_shopping_item

# from app.voice.swear_jar import check_profanity

# from app.voice.swear_jar import remove_swear_intent

# from app.voice.time import schedule_meeting_intent

# from app.voice.time import latest_time_up

# from app.voice.basic_commands import audio_recorder, \
#     dismiss_ai, \
#     question_time, \
#     play_news, \
#     set_alarm_reminder, \
#     weather_query, \
#     random_intent, \
#     play_music

spacy_tagger = spacy.load('en_core_web_sm')

default_question_answering = pipeline('question-answering')

BOT_NAME = 'cain'


def get_shopping_classification(text):
    # pred, prob = find_shopping_item(text)
    # return pred, prob

    pass


def quick_predict_label(text, cutoff=0.0):
    for skill in UserSkills.query.all():
        skill_name = skill.skill_name
        if skill_name not in global_skills_dic:
            log.debug("Warning unavailable skill but exists in database")
        global_skills_dic[skill.name].handle_classification(text)

    # print("[DEBUG] First layer general prediction", predicted_labels, prob)
    # predict_shopping, shopping_prob = get_shopping_classification(text)
    # print("[DEBUG] Shopping layer general prediction", predicted_labels, prob)
    # includes_swear, swear_count = check_profanity(text)
    # print("[DEBUG] Swear detection: ", includes_swear, swear_count)

    # print("Moving to regular skills")
    # print("[DEBUG] Clear Swear detection: ", remove_swear_intent(text))
    # print("[DEBUG] Schedule Meeting Detection: ", schedule_meeting_intent(text))
    # print("[DEBUG] Latest Time up Detection: ", latest_time_up(text))

    # print("[DEBUG] Record Audio: ", audio_recorder(text))
    # print("[DEBUG] Dismiss AI: ", dismiss_ai(text))
    # print("[DEBUG] What time/day is it?: ", question_time(text))
    # print("[DEBUG] Play news?: ", play_news(text))
    # print("[DEBUG] Get weather info?: ", weather_query(text))
    # print("[DEBUG] Should set alarm?: ", set_alarm_reminder(text))
    # print("[DEBUG] Play music?: ", play_music(text))
    # print("[DEBUG] Should give randomized number?: ", random_intent(text))
    # if shopping_prob > prob:
    #     return "Shopping; " + predict_shopping, shopping_prob
    # return predicted_labels[0], prob[0]

    pass


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
