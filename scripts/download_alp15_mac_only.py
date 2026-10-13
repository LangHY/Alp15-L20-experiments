"""仅在Mac下载固定949候选clip：不连接SSH、不上传。"""
import argparse
import sys

import download_alp15_mac_l20_pipeline as engine


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', '--run', action='store_true')
    parser.add_argument('--confirm-bytes', type=int)
    parser.add_argument('--output', default=engine.DEFAULT_LOCAL)
    args = parser.parse_args(argv)
    forwarded = ['--download-only', '--local-dir', args.output]
    if args.download:
        forwarded.append('--run')
    if args.confirm_bytes is not None:
        forwarded += ['--confirm-bytes', str(args.confirm_bytes)]
    return engine.main(forwarded)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        safe = isinstance(exc, (engine.PipelineError, engine.core.DownloadError))
        message = str(exc) if safe else type(exc).__name__
        print('STOP: ' + message + '; files/cache preserved.', file=sys.stderr)
        sys.exit(1)
