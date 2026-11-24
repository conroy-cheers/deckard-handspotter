"""Review and label synchronized Steam Frame recordings and HOT3D clips."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import secrets
import sys
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import urlsplit

from model_jobs import ModelJobs
from workspace import CAMERAS, HANDS, Workspace, image_name, ingest


def serve(workspace, port, desktop_token=None):
    editor = Path(__file__).parent
    lock = Lock()
    undo = {}
    jobs = ModelJobs(workspace, Path(__file__).resolve())

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, payload, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def json(self, payload, status=200):
            self.send_bytes(json.dumps(payload, allow_nan=False).encode(), "application/json", status)

        def local(self):
            host = self.headers.get("Host")
            if host not in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"):
                self.send_error(403)
                return False
            if desktop_token and not secrets.compare_digest(
                self.headers.get("X-Labeler-Token", "").encode(), desktop_token
            ):
                self.send_error(403)
                return False
            return True

        def do_GET(self):
            if not self.local():
                return
            path = urlsplit(self.path).path
            if path in ("/", "/editor.js", "/editor.css"):
                source, mime = {
                    "/": ("index.html", "text/html"),
                    "/editor.js": ("editor.js", "text/javascript"),
                    "/editor.css": ("editor.css", "text/css"),
                }[path]
                self.send_bytes((editor / source).read_bytes(), mime)
                return
            if path == "/api/workspace":
                self.json(dict(take=workspace.manifest["take"], cameras=workspace.manifest["cameras"],
                               groups=[{key: group[key] for key in ("sequence", "time_us")}
                                       for group in workspace.groups],
                               order=workspace.cameras, hands=HANDS, labels=workspace.load_labels(),
                               proposals=workspace.load_proposals(), terra=workspace.load_terra(),
                               model_capabilities=jobs.capabilities))
                return
            if path == "/api/model":
                self.json(jobs.status())
                return
            match = path.removeprefix("/frames/").split("/") if path.startswith("/frames/") else []
            if len(match) == 2 and match[0] in workspace.cameras and match[1].endswith(".png") and match[1][:-4].isdigit():
                sequence = int(match[1][:-4])
                if sequence in workspace.sequences:
                    self.send_bytes((workspace.root / image_name(sequence, match[0])).read_bytes(), "image/png")
                    return
            self.send_error(404)

        def do_POST(self):
            if not self.local():
                return
            if self.path not in ("/api/label", "/api/undo", "/api/model", "/api/model/stop"):
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                origin = self.headers.get("Origin")
                if (self.headers.get_content_type() != "application/json" or not 0 < length <= 4096
                        or origin not in (None, f"http://{self.headers['Host']}")):
                    raise ValueError("Expected a local JSON label request")
                data = json.loads(self.rfile.read(length))
                if self.path.startswith("/api/model"):
                    if self.path == "/api/model":
                        jobs.start(data)
                    else:
                        jobs.stop()
                    self.json(jobs.status())
                    return
                with lock:
                    key = workspace.key(data["sequence"], data["camera"], data["hand"])
                    if self.path == "/api/undo":
                        entry = undo.get(key)
                        if not entry or entry[0] != data["token"]:
                            raise ValueError("This edit can no longer be undone. Reload labels.")
                        _, previous, expected = entry
                        workspace.undo_edit(key, previous, expected)
                        del undo[key]
                        result = {"label": previous}
                    else:
                        previous, label = workspace.edit_label(**data)
                        token = secrets.token_urlsafe(18)
                        undo[key] = (token, previous, label)
                        result = {"label": label, "token": token}
                self.json(result)
            except (KeyError, OSError, TypeError, ValueError) as error:
                self.json({"error": str(error)}, 400)

        def log_request(self, code="-", size="-"):
            if str(code).startswith(("4", "5")):
                super().log_request(code, size)

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        if desktop_token:
            def stop_on_close():
                sys.stdin.buffer.read()
                server.shutdown()

            Thread(target=stop_on_close, daemon=True).start()
        print(f"Open http://127.0.0.1:{server.server_port} to label {workspace.manifest['take']}", flush=True)
        try:
            server.serve_forever()
        finally:
            jobs.close()


def main():
    parser = argparse.ArgumentParser(prog="frame-hand-labeler", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("ingest", help="Import synchronized Frame or HOT3D camera views")
    prepare.add_argument("take", type=Path)
    prepare.add_argument("workspace", type=Path)
    prepare.add_argument("--start", type=float, default=0, help="Trim to this clip time in seconds")
    prepare.add_argument("--duration", type=float, help="Clip duration in seconds")
    cadence = prepare.add_mutually_exclusive_group()
    cadence.add_argument("--every", type=int, default=1, help="Keep one group out of every N")
    cadence.add_argument("--hz", type=float, help="Keep the nearest complete group to each time step")
    prepare.add_argument("--channel-map", action="append", metavar="SOURCE=FRAME_CAMERA",
                         help="Select and name a HOT3D stream. Repeat to map multiple streams.")
    prepare.add_argument("--warp-rig", type=Path, metavar="RIG_JSON",
                         help="Warp HOT3D tracking views into this Frame front-camera calibration")
    editor = commands.add_parser("serve", help="Open the local box annotation and review workspace")
    editor.add_argument("workspace", type=Path)
    editor.add_argument("--port", type=int, default=8765)
    predict = commands.add_parser("propagate", help="Propose boxes from manual seeds using SAM 3.1")
    predict.add_argument("workspace", type=Path)
    predict.add_argument("--camera", choices=CAMERAS, action="append", help="Camera(s) to track. Defaults to all imported views.")
    predict.add_argument("--radius", type=int, default=30, help="Maximum distance in sampled groups from a seed")
    predict.add_argument("--start-index", type=int, default=0, help="First target group, zero-based")
    predict.add_argument("--limit", type=int, help="Maximum number of consecutive target groups")
    predict.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    predict.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    predict.add_argument("--weights", default=os.getenv("FRAME_HAND_LABELER_WEIGHTS"),
                         help="Override the SAM 3.1 Multiplex safetensors checkpoint")
    review = commands.add_parser("terra", help="Review sampled groups with GPT-6.1 Terra via Codex")
    review.add_argument("workspace", type=Path)
    review.add_argument("--start-index", type=int, default=0, help="Start at this zero-based group index")
    review.add_argument("--limit", type=int, help="Review at most this many groups from the start index")
    review.add_argument("--hz", type=float, help="Target review rate using available workspace frames")
    review.add_argument("--groups-per-request", type=int, default=4)
    review.add_argument("--workers", type=int, default=3)
    resolve = commands.add_parser("resolve-hands", help="Reconcile Terra hand identities across cameras and time")
    resolve.add_argument("workspace", type=Path)
    resolve.add_argument("--xrservice", type=Path, default=Path("/persist/xrservice.json"))
    resolve.add_argument("--device-config", type=Path, default=Path("/persist/device_config.json"))
    resolve.add_argument("--anchors", type=Path, action="append", default=[],
                         help="Workspace for this recording with trusted hand boxes, repeatable")
    contact = commands.add_parser("review-contacts", help="Flag touching hands and review those spans with Terra")
    contact.add_argument("workspace", type=Path)
    contact.add_argument("--xrservice", type=Path, default=Path("/persist/xrservice.json"))
    contact.add_argument("--device-config", type=Path, default=Path("/persist/device_config.json"))
    contact.add_argument("--workers", type=int, default=4)
    contact.add_argument("--plan-only", action="store_true", help="Count flagged groups without requesting review")
    export = commands.add_parser("export", help="Export reviewed labels as grouped JSON and COCO")
    export.add_argument("workspace", type=Path)
    export.add_argument("--output", type=Path, help="Default is WORKSPACE/labels")
    export.add_argument("--include-terra", action="store_true", help="Include model labels where there is no human edit")
    args = parser.parse_args()

    try:
        if args.command == "ingest":
            if args.take.is_file():
                from hot3d_input import ingest_hot3d

                n = ingest_hot3d(args.take, args.workspace, args.start, args.duration, args.every,
                                 hz=args.hz, channels=args.channel_map, rig=args.warp_rig)
            else:
                if args.channel_map or args.warp_rig:
                    raise ValueError("--channel-map and --warp-rig require a HOT3D clip archive")
                n = ingest(args.take, args.workspace, args.start, args.duration, args.every, hz=args.hz)
            print(f"Imported {n} synchronized groups to {args.workspace}")
            return
        workspace = Workspace(args.workspace)
        if workspace.is_hot3d and args.command in ("resolve-hands", "review-contacts"):
            raise ValueError(f"{args.command} requires native Steam Frame camera geometry")
        if args.command == "serve":
            token = os.environ.get("FRAME_HAND_LABELER_DESKTOP_TOKEN")
            serve(workspace, args.port, token.encode() if token else None)
        elif args.command == "propagate":
            from tracker import propagate

            n, old = propagate(workspace, args.camera, args.radius, args.device, args.threads, args.weights,
                               start=args.start_index, limit=args.limit)
            print(f"Saved {n} proposals, replacing {old} previous proposals")
        elif args.command == "terra":
            from terra import review as review_clip

            count = review_clip(workspace, args.start_index, args.limit, args.groups_per_request, args.workers,
                                hz=args.hz)
            print(f"Saved {count} new Terra reviews")
        elif args.command == "review-contacts":
            from contact import review_contacts
            from terra import review_lock

            with review_lock(workspace):
                review_contacts(workspace, args.xrservice, args.device_config, args.workers, plan_only=args.plan_only)
        elif args.command == "resolve-hands":
            from handedness import reconcile

            report = reconcile(workspace, args.xrservice, args.device_config, list(map(Workspace, args.anchors)))
            print(f"Swapped {report['swapped_views']} of {report['views']} views with hand boxes "
                  f"using {report['associations']} calibrated or temporal associations")
        else:
            output = args.output or args.workspace / "labels"
            grouped, images, boxes = workspace.export_counts(output, include_terra=args.include_terra)
            print(f"Grouped training labels (boxes): {grouped} in {output / 'groups.json'}")
            print(f"COCO (fully reviewed views: {images}, boxes: {boxes}) in {output / 'coco.json'}")
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError) as error:
        parser.exit(1, f"frame-hand-labeler: {error}\n")


if __name__ == "__main__":
    main()
