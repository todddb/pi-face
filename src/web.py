# src/web.py
from __future__ import annotations

import os
import shutil
from pathlib import Path

from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from .config import BASE_DIR, load_config
from .db import KnownFaceEncoding, Person, UnknownFace, get_session, init_db


cfg = load_config()
init_db(cfg.database.url)

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

app.secret_key = os.environ.get("PI_FACE_SECRET_KEY", "dev-secret-key")


@app.route("/")
def index():
    return redirect(url_for("people"))


@app.route("/people", methods=["GET", "POST"])
def people():
    with get_session() as session:
        if request.method == "POST":
            first = request.form.get("first_name", "").strip()
            last = request.form.get("last_name", "").strip() or None
            email = request.form.get("email", "").strip() or None
            vip = request.form.get("vip") == "on"
            ignore = request.form.get("ignore") == "on"

            if not first:
                flash("First name is required", "error")
            else:
                person = Person(
                    first_name=first,
                    last_name=last,
                    email=email,
                    vip=vip,
                    ignore=ignore,
                )
                session.add(person)
                session.commit()
                flash("Person created", "success")

        people = session.query(Person).order_by(Person.created_at.desc()).all()

    return render_template("people.html", people=people)


@app.route("/unrecognized", methods=["GET"])
def unrecognized():
    with get_session() as session:
        faces = (
            session.query(UnknownFace)
            .order_by(UnknownFace.detected_at.desc())
            .limit(100)
            .all()
        )
        people = session.query(Person).order_by(Person.first_name.asc()).all()

    return render_template("unrecognized.html", faces=faces, people=people)


@app.route("/unrecognized/<int:face_id>/assign", methods=["POST"])
def assign_unrecognized(face_id: int):
    person_id = request.form.get("person_id")
    if not person_id:
        flash("Select a person to assign", "error")
        return redirect(url_for("unrecognized"))

    try:
        person_id_int = int(person_id)
    except ValueError:
        flash("Invalid person selection", "error")
        return redirect(url_for("unrecognized"))

    with get_session() as session:
        face = session.get(UnknownFace, face_id)
        person = session.get(Person, person_id_int)

        if face is None or person is None:
            flash("Face or person not found", "error")
            return redirect(url_for("unrecognized"))

        source_path = BASE_DIR / face.image_path

        known_dir = Path(cfg.storage.known_faces_dir)
        if known_dir.is_absolute():
            try:
                known_rel = known_dir.relative_to(BASE_DIR)
            except ValueError:
                known_rel = Path("data/known_faces")
        else:
            known_rel = known_dir

        target_rel = known_rel / Path(face.image_path).name
        target_path = (BASE_DIR / target_rel).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.move(str(source_path), str(target_path))
        except FileNotFoundError:
            # Continue even if the file is missing; DB state is still updated
            flash("Source image missing; only updating database", "warning")

        encoding = face.encoding
        encoding_dim = face.encoding_dim or 128

        session.add(
            KnownFaceEncoding(
                person_id=person.id,
                image_path=str(target_rel),
                encoding=encoding,
                encoding_dim=encoding_dim,
            )
        )

        session.delete(face)
        session.commit()

    flash("Face assigned to person", "success")
    return redirect(url_for("unrecognized"))


@app.route("/known-faces", methods=["GET"])
def known_faces():
    with get_session() as session:
        encodings = (
            session.query(KnownFaceEncoding)
            .order_by(KnownFaceEncoding.created_at.desc())
            .all()
        )
        people_lookup = {p.id: p for p in session.query(Person).all()}

    return render_template("known_faces.html", encodings=encodings, people=people_lookup)


@app.route("/known-faces/<int:encoding_id>/delete", methods=["POST"])
def delete_known_face(encoding_id: int):
    with get_session() as session:
        enc = session.get(KnownFaceEncoding, encoding_id)
        if enc is None:
            flash("Encoding not found", "error")
            return redirect(url_for("known_faces"))

        image_path = BASE_DIR / enc.image_path
        session.delete(enc)
        session.commit()

    try:
        image_path.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass

    flash("Encoding removed", "success")
    return redirect(url_for("known_faces"))


@app.route("/images/<path:filename>")
def serve_image(filename: str):
    candidate = (BASE_DIR / filename).resolve()
    if BASE_DIR not in candidate.parents and candidate != BASE_DIR:
        abort(404)
    if not candidate.exists():
        abort(404)
    relative = candidate.relative_to(BASE_DIR)
    return send_from_directory(BASE_DIR, str(relative))


def run():  # pragma: no cover - convenience entrypoint
    app.run(host="0.0.0.0", port=8000, debug=True)


if __name__ == "__main__":  # pragma: no cover
    run()
