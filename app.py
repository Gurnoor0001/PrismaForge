from utils.utils import adaptive_instance_normalization
from utils.models import VGGEncoder, Decoder
from unittest import result
from torch import device
from sympy import content
import os
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import InputRequired
from PIL import Image
from torchvision import transforms
import io


app = Flask(__name__)
app.config['SECRET_KEY'] = "key"
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['UPLOAD_FOLDER'] = "static/uploads"
app.config['ALLOWED_EXTENSIONS'] = ['png','jpg','jpeg']
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class UploadForm(FlaskForm):
    content = FileField("Content Image")
    style = FileField("Style Image")
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField("Alpha",default=1.0)
    submit = SubmitField("Transfer Style")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = VGGEncoder(r"C:\Major projects\NST-Neural Style Transfer\vgg_normalised.pth").to(device)
decoder = Decoder().to(device)
state_dict = torch.load(r"C:\Major projects\NST-Neural Style Transfer\decoder_final.pth")
for k in list(state_dict.keys()):
    if k.startswith('net.'):
        state_dict[k.replace('net.', 'decoder.')] = state_dict.pop(k)
decoder.load_state_dict(state_dict)

encoder.eval()
decoder.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in app.config['ALLOWED_EXTENSIONS']  

def style_transfer(content_img,style_img,alpha,resolution,encoder,decoder,device):

    res = int(resolution)

    content_transform = transforms.Compose([
        transforms.Resize(res),
        transforms.ToTensor(),

    ])

    style_transform = transforms.Compose([
        transforms.Resize(res),
        transforms.ToTensor(),
       
    ])

    content_img = content_transform(content_img).unsqueeze(0).to(device)
    style_img = style_transform(style_img).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(content_img, is_test = True)
        style_feats = encoder(style_img, is_test = True)
        stylized_feats = adaptive_instance_normalization(content_feats,style_feats)

        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats
        
        styled_img = decoder(stylized_feats)

    return styled_img


def save_img(tensor,filepath):

    tensor = tensor.cpu().clone()
    tensor = tensor.squeeze(0)
    tensor = tensor.clamp(0 , 1) 
    image = transforms.ToPILImage(mode="RGB")(tensor)
    image.save(filepath)
    

@app.route("/", methods=["GET","POST"])
def index():
    form = UploadForm()
    result_img = None
    content_filename = None 
    style_filename = None 
    error = None

    if form.validate_on_submit():
        if form.content.data and form.content.data.filename:
            if allowed_file(form.content.data.filename):
                content_filename = secure_filename(form.content.data.filename) 
                form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'],content_filename))
                form.content_path.data = content_filename
        else:
            content_filename = form.content_path.data

        if form.style.data and form.style.data.filename:
            if allowed_file(form.style.data.filename):
                style_filename = secure_filename(form.style.data.filename)
                form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'],style_filename))
                form.style_path.data = style_filename
        else:
            style_filename = form.style_path.data

        if content_filename and style_filename:
                content_path = os.path.join(app.config['UPLOAD_FOLDER'],content_filename)
                style_path = os.path.join(app.config['UPLOAD_FOLDER'],style_filename)
                try :
                    content_img = Image.open(content_path).convert("RGB")
                    style_img = Image.open(style_path).convert("RGB")

                    alpha = float(form.alpha.data)
                    res = form.resolution.data
                    
                    styleized_img = style_transfer(content_img,style_img,alpha,res, encoder, decoder, device)

                    result_filename = "stylized" + content_filename

                    result_path = os.path.join(app.config['UPLOAD_FOLDER'],result_filename)

                    save_img(styleized_img, result_path)

                    result_img = result_filename
                    

                 
                except Exception as e:
                    error = str(e)
                
    elif request.method == 'POST':
        if not content_filename:
            error = "Upload content image"
        elif not style_filename:
            error = "Upload style image"
        
    return render_template("index.html",form = form, result_image = result_img, content_image = content_filename,style_image = style_filename, error = error)

@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)


@app.route("/examples/<path:filename>")
def send_example(filename):
    return send_from_directory("examples",filename)

    
if __name__ == "__main__":
    from werkzeug.serving import run_simple
    run_simple("localhost", 5000, app, use_debugger=True, use_reloader=True)
    
    
  
