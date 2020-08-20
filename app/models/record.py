from sqlalchemy.orm import relationship

from app import db
import datetime
import requests


class Record(db.Model):
    __table_args__ = {'extend_existing': True}
    record_id = db.Column(db.Integer, primary_key=True, unique=True, autoincrement=True)
    log = db.Column(db.String(140))
    intent_classification = db.Column(db.String(140))
    created_date = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    tags = relationship("Tag")


class Tag(db.Model):
    __table_args__ = {'extend_existing': True}
    extend_existing = True
    record_id = db.Column(db.Integer, db.ForeignKey('record.record_id'))
    tag_id = db.Column(db.Integer, primary_key=True, unique=True, autoincrement=True)
    keyword = db.Column(db.String)
    pp_speech = db.Column(db.String)
    record = relationship('Record')


class User(db.Model):
    __table_name__ = 'user'
    __table_args__ = {'extend_existing': True}
    user_id = db.Column(db.Integer, primary_key=True, unique=True)
    skills_registed = db.Column(db.Integer, db.ForeignKey('skill.skill_id'))
    skills = relationship('Skills', uselist=True)


class Skills(db.Model):
    __table_name__ = 'skills'
    __table_args__ = {'extend_existing': True}
    skill_id = db.Column(db.Integer, primary_key=True, unique=True)
    skill_name = db.Column(db.String)
    skill_description = db.Column(db.String)
    skill_endpoint = db.Column(db.String)

    @static_method
    def register_skill(skill_name, skill_description, skill_endpoint):
        new_skill = Skills(skill_name=skill_name,skill_description=skill_description)
        endpoint_response = requests.post(skill_endpoint, json={'message': "test"})
        if endpoint_response.status_code != 200:
            return ("Failed to Register", 400)
        
        new_skill.skil_endpoint = skill_endpoint
        db.session.add(skil_endpoint)
        db.session.commit()
        return ("Successfully Registered", 200)