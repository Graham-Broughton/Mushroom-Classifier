from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired

class ImageForm(FlaskForm):
    image = FileField(
        'image',
        validators=[FileRequired(message="Please include 'image' field.")
        ])
    submit = SubmitField('Upload')

class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos, 'Only images allowed'),
            FileRequired('File field should not be empty')
        ]
    )
    submit = SubmitField('Upload')