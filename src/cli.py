import argparse
from src.ingest import pdf_parser, excel_reader
from src.aggregator.consolidate import consolidate
from src.exporters.excel_writer import write_excel


def main():
    p = argparse.ArgumentParser(description="Exam reconciler CLI")
    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("ingest-pdf")
    sp.add_argument("path")

    sp2 = sub.add_parser("reconcile")
    sp2.add_argument("input")
    sp2.add_argument("output")

    args = p.parse_args()
    if args.cmd == "ingest-pdf":
        count = pdf_parser.ingest_pdf(args.path)
        print(f"imported {count} entries into database")
    elif args.cmd == "reconcile":
        rows = excel_reader.read_excel(args.input)
        out = consolidate(rows)
        write_excel(out, args.output)
        print(f"wrote {len(out)} rows to {args.output}")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
