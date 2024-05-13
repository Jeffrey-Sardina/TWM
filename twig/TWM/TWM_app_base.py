# adopted from https://www.delftstack.com/howto/python-flask/flask-display-image/
# and also from: https://www.geeksforgeeks.org/retrieving-html-from-data-using-flask/
# and https://www.w3schools.com/tags/tag_select.asp

# importing Flask and other modules
from flask import Flask, request, render_template, redirect
import os
from draw_twm import do_twm, load_hyps_dict

# Flask constructor
app = Flask(__name__) 

IMG_FOLDER = os.path.join("static", "images")
app.config["UPLOAD_FOLDER"] = IMG_FOLDER
hyp_selected = {}

# A decorator used to tell the application
# which URL is associated function
@app.route('/', methods =["GET", "POST"])
def twm_demo():
    global hyp_selected
    if request.method == "POST":
        err = ""
        try:
            try:
                margin = request.form.get("margin")
                margin = float(margin)
                if request.form.get("loss") != "MarginRankingLoss":
                    err += "margin parameter can only be given when MarginRankingLoss is also given\n"
                    assert False, err
            except:
                if margin == 'None':
                    if request.form.get("loss") == "MarginRankingLoss":
                        err += "margin parameter must be set when MarginRankingLoss is given\n"
                        assert False, err
                    margin = None
            use_TWIG = request.form.get("use_twig")
            use_TWIG = True if use_TWIG == "Yes" else False
            dataset = request.form.get("dataset")
            hyp_selected = {
                "loss": request.form.get("loss"),
                "neg_samp": request.form.get("negsamp"),
                "lr":float( request.form.get("lr")),
                "reg_coeff": float(request.form.get("reg_coeff")),
                "npp": int(request.form.get("npp")),
                "margin": margin,
                "dim": int(request.form.get("dim"))
            }
        except:
            img = os.path.join(app.config["UPLOAD_FOLDER"], "temp.jpg")
            twm_text = f'Please fill in the whole form or else enjoy some LoTR memes!\n'
            twm_text += f'Specific errors follow:\n{err}'
            return render_template(
                "index.html",
                twm_image=img,
                twm_text=twm_text,
            )
        img, graph_save_url, mrr_pred, exp_id_wanted = do_twm(
            dataset,
            hyp_selected,
            hyps_dict,
            use_TWIG=use_TWIG
        )
        img = os.path.join(app.config["UPLOAD_FOLDER"], img)
        if use_TWIG:
            twm_text = f'Predicted MRR: {str(round(float(mrr_pred), 3)).ljust(5, "0")}'
        else:
            twm_text = f'Ground Truth MRR: {str(round(float(mrr_pred), 3)).ljust(5, "0")}'
        hyp_text = f'Hyperparameter grid iD: {exp_id_wanted}'
        link_label = f'Graph URL accession: '
        link = f'{graph_save_url}'
        return render_template(
            "index.html",
            twm_image=img,
            twm_text=twm_text,
            hyp_text=hyp_text,
            link_label=link_label,
            link=link
        )
    else:
        img = os.path.join(app.config["UPLOAD_FOLDER"], "temp.jpg")
        twm_text = f'Enjoy some LoTR memes!'
        return render_template(
            "index.html",
            twm_image=img,
            twm_text=twm_text
        )

if __name__ == "__main__":
    hyps_dict = load_hyps_dict("hyp.grid")
    app.run(debug=True)
