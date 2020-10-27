from sqlalchemy.orm import relationship

from app import db
import datetime


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


class UserSkills(db.Model):
    __table_args__ = {'extend_existing': True}
    user_skill_id = db.Column(db.Integer, primary_key=True, unique=True, autoincrement=True)
    skill_name = db.Column(db.String)
    activated = db.Boolean()

