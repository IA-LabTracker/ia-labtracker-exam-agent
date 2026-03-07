from __future__ import annotations

import argparse

from src.utils.logging import logger


def cmd_migrate(_args: argparse.Namespace) -> None:
    from src.db.client import DBClient

    db = DBClient().connect()
    db.run_migrations()
    db.close()
    logger.info("Migrations applied successfully")


def cmd_ingest_pdf(args: argparse.Namespace) -> None:
    from src.db.client import DBClient
    from src.embeddings.embedder import Embedder
    from src.ingest.pdf_parser import extract_questions, file_hash

    db = DBClient().connect()
    embedder = Embedder.create()

    fhash = file_hash(args.file)
    if db.file_already_ingested(fhash):
        logger.info("File already ingested, skipping: %s", args.file)
        return

    questions = extract_questions(
        args.file, institution=args.institution, year=args.year
    )
    count = db.upsert_questions(questions)

    pending = db.get_questions_without_embeddings()
    if pending:
        texts = [
            f"{q['tema_normalized'] or ''} {q['subtema_normalized'] or ''} {q['raw_text']}"
            for q in pending
        ]
        embeddings = embedder.embed_batch(texts)
        for q, emb in zip(pending, embeddings):
            db.update_embedding(q["id"], emb)

    db.record_ingest(args.file, fhash, count)
    db.close()
    logger.info("Ingested %d questions from %s", count, args.file)


def cmd_ingest_stats(args: argparse.Namespace) -> None:
    from src.db.client import DBClient
    from src.embeddings.embedder import Embedder
    from src.ingest.pdf_parser import extract_theme_stats, file_hash

    db = DBClient().connect()
    embedder = Embedder.create()

    fhash = file_hash(args.file)
    if db.file_already_ingested(fhash):
        logger.info("File already ingested, skipping: %s", args.file)
        return

    stats = extract_theme_stats(args.file, institution=args.institution)
    count = db.upsert_theme_stats(stats)

    _generate_theme_stats_embeddings(db, embedder)

    db.record_ingest(args.file, fhash, count)
    db.close()
    logger.info("Ingested %d theme stats from %s", count, args.file)


def _generate_theme_stats_embeddings(db, embedder) -> None:
    pending = db.get_theme_stats_without_embeddings()
    if not pending:
        return
    texts = [f"{s['tema']} {s['subtema'] or ''}" for s in pending]
    embeddings = embedder.embed_batch(texts)
    for s, emb in zip(pending, embeddings):
        db.update_theme_stat_embedding(s["id"], emb)
    logger.info("Generated embeddings for %d theme_stats entries", len(pending))


def cmd_reconcile(args: argparse.Namespace) -> None:
    from src.aggregator.consolidate import reconcile_all, reverse_coverage
    from src.db.client import DBClient
    from src.embeddings.embedder import Embedder
    from src.exporters.excel_writer import write_excel
    from src.ingest.excel_reader import read_excel

    db = DBClient().connect()
    embedder = Embedder.create()

    # Ensure theme_stats have embeddings
    _generate_theme_stats_embeddings(db, embedder)

    input_rows = read_excel(args.input)
    results = reconcile_all(input_rows, embedder, db)
    reverse_rows = reverse_coverage(results, embedder, db)
    write_excel(results, args.output, also_csv=args.csv, reverse_rows=reverse_rows)

    db.close()
    logger.info("Output written to %s", args.output)


def main() -> None:
    parser = argparse.ArgumentParser(prog="exam-cli", description="Exam Reconciler CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("migrate")

    p_ingest = sub.add_parser("ingest-pdf")
    p_ingest.add_argument("--file", required=True)
    p_ingest.add_argument("--institution", default="unknown")
    p_ingest.add_argument("--year", type=int, default=None)

    p_stats = sub.add_parser("ingest-stats")
    p_stats.add_argument("--file", required=True)
    p_stats.add_argument("--institution", default="unknown")

    p_rec = sub.add_parser("reconcile")
    p_rec.add_argument("--input", required=True)
    p_rec.add_argument("--output", default="ranking_output.xlsx")
    p_rec.add_argument("--csv", action="store_true")

    args = parser.parse_args()

    commands = {
        "migrate": cmd_migrate,
        "ingest-pdf": cmd_ingest_pdf,
        "ingest-stats": cmd_ingest_stats,
        "reconcile": cmd_reconcile,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
