#!/usr/bin/env python3

import os
import re
import subprocess
import threading

METEOR_JAR = "meteor-1.5.jar"


def clean_meteor_text(text):
    """Make text safe for METEOR's one-line stdio protocol."""
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)

    text = text.replace("|||", " ")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "empty"


class Meteor:
    """Python wrapper for METEOR 1.5 using corpus-level EVAL."""

    def __init__(self):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        jar_path = os.path.join(module_dir, METEOR_JAR)

        if not os.path.isfile(jar_path):
            raise FileNotFoundError(
                f"Cannot find {METEOR_JAR}. Expected it at: {jar_path}"
            )

        self.meteor_cmd = [
            "java",
            "-Xmx2G",
            "-jar",
            jar_path,
            "-",
            "-",
            "-stdio",
            "-l",
            "en",
            "-norm",
        ]

        self.meteor_p = subprocess.Popen(
            self.meteor_cmd,
            cwd=module_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self.lock = threading.Lock()

    def _check_process(self):
        return_code = self.meteor_p.poll()
        if return_code is not None:
            stderr_text = self.meteor_p.stderr.read()
            raise RuntimeError(
                "METEOR Java process terminated unexpectedly "
                f"with code {return_code}. Stderr: {stderr_text}"
            )

    def _write_line(self, line):
        self._check_process()
        self.meteor_p.stdin.write(line + "\n")
        self.meteor_p.stdin.flush()

    def _read_line(self, description):
        line = self.meteor_p.stdout.readline()
        if line == "":
            self._check_process()
            stderr_text = self.meteor_p.stderr.read()
            raise RuntimeError(
                f"METEOR returned no {description}. Stderr: {stderr_text}"
            )
        return line.strip()

    def _stat_unlocked(self, hypothesis_str, reference_list):
        hypothesis = clean_meteor_text(hypothesis_str)
        references = [clean_meteor_text(ref) for ref in reference_list]

        score_line = " ||| ".join(
            ("SCORE", " ||| ".join(references), hypothesis)
        )
        self._write_line(score_line)
        return self._read_line("sufficient-statistics line")

    def _stat(self, hypothesis_str, reference_list):
        """Return METEOR sufficient statistics for one paired example."""
        with self.lock:
            return self._stat_unlocked(hypothesis_str, reference_list)

    def compute_score(self, gts, res):
        """
        Compute one corpus-level METEOR score and per-example scores.

        gts: dict[id, list[str]] containing one or more references per id.
        res: dict[id, list[str]] containing exactly one hypothesis per id.
        """
        if gts.keys() != res.keys():
            raise ValueError("gts and res must contain exactly the same keys.")

        image_ids = list(gts.keys())
        if not image_ids:
            return 0.0, []

        with self.lock:
            statistics = []
            for image_id in image_ids:
                if len(res[image_id]) != 1:
                    raise ValueError(
                        f"Expected one hypothesis for {image_id!r}; "
                        f"received {len(res[image_id])}."
                    )
                if len(gts[image_id]) < 1:
                    raise ValueError(
                        f"Expected at least one reference for {image_id!r}."
                    )

                statistics.append(
                    self._stat_unlocked(res[image_id][0], gts[image_id])
                )

            eval_line = "EVAL ||| " + " ||| ".join(statistics)
            self._write_line(eval_line)

            segment_scores = [
                float(self._read_line("per-example score"))
                for _ in image_ids
            ]
            corpus_score = float(self._read_line("corpus score"))

        return corpus_score, segment_scores

    def _score(self, hypothesis_str, reference_list):
        """Compute METEOR for one example; retained for API compatibility."""
        with self.lock:
            statistics = self._stat_unlocked(hypothesis_str, reference_list)
            self._write_line(f"EVAL ||| {statistics}")

            self._read_line("single-example intermediate score")
            final_score = float(self._read_line("single-example final score"))

        return final_score

    def method(self):
        return "METEOR"

    def close(self):
        process = getattr(self, "meteor_p", None)
        if process is None:
            return

        with self.lock:
            if process.poll() is None:
                try:
                    process.stdin.close()
                except Exception:
                    pass
                process.kill()
                process.wait()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
