"""Portable dashboard launcher. Default is offline archives only; Python stdlib."""
import argparse,json,time,urllib.parse
from pathlib import Path
from http.server import ThreadingHTTPServer
from server import Repository,make_handler,read_json

class OfflineRepository(Repository):
    def __init__(self,root):
        super().__init__(root)
        self.frozen=read_json(root/'dashboard/cache/last_server_snapshot.json',{}).get('experiments',[])
        for path in (root/'compression/results').glob('*/recorded_metrics.json'):
            record=read_json(path,{})
            if isinstance(record,dict) and record.get('id') and 'task_validation' in record:self.frozen.append(record)
    def get(self):
        local=super().get()
        items={e['id']:dict(e,source='local_archive') for e in self.frozen}
        items.update({e['id']:dict(e,source='local_archive') for e in local['experiments']})
        # Explicit exports supplement experiments for which native server files were not retrieved.
        for e in self.frozen:
            old=items.get(e['id'])
            if old is None or (e.get('current_step',0)>=old.get('current_step',0) and e.get('status')=='completed'):
                items[e['id']]=dict(e,source='local_archive')
        result=dict(local);result.update(experiments=sorted(items.values(),key=lambda e:e.get('started_at') or e['id'],reverse=True),connection='local',connection_error=None)
        return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--remote',help='Optional existing loopback SSH-forwarded server API, e.g. http://127.0.0.1:18766')
    args=parser.parse_args();root=Path(__file__).resolve().parent.parent
    if args.remote and urllib.parse.urlsplit(args.remote).hostname not in ('127.0.0.1','localhost'):parser.error('Remote must be a loopback SSH forward')
    repo=Repository(root,args.remote) if args.remote else OfflineRepository(root)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(repo))
    print('Dashboard: http://127.0.0.1:'+str(args.port)+' ('+('SSH API' if args.remote else 'offline archives; no server requests')+')',flush=True)
    server.serve_forever()

if __name__=='__main__':main()
