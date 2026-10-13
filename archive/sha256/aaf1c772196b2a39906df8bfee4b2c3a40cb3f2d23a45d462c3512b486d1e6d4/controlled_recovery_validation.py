"""Frozen twenty-clip task validation using independently seeded trajectory noise."""
import json
import statistics
from pathlib import Path
import torch
from controlled_8b_diagnostic import generate_prefix, sampled, digest
from run_mlp_experiment import file_digest, write_json
from student_interface_distill import exported_weights, mlp_tensor_digest


class ControlledValidation:
    def __init__(self, directory, output, provenance, split, parent_experiment=None,check_parent_reproduction=True):
        self.directory, self.output = Path(directory), Path(output)
        self.check_parent_reproduction=check_parent_reproduction
        self.reference = json.loads((self.directory/'result.json').read_text())
        assert self.reference['status'] == 'completed'
        for key in ('source_commit', 'checkpoint_revision', 'dataset_revision'):
            assert self.reference[key] == provenance[key]
        self.seeds = self.reference['trajectory_seeds']
        assert self.seeds == [42,43,44]
        self.rows = [r for r in self.reference['records'] if r['model']=='dense' and r['prefix']=='teacher']
        self.clips = sorted({r['clip_id'] for r in self.rows})
        assert len(self.clips)==20 and len(self.rows)==60
        assert not set(self.clips)&set(sum((split[k] for k in ('calibration','validation','holdout')),[]))
        self.dense = {(r['clip_id'],r['seed']):r for r in self.rows}
        self.parent_reference = None
        parent_file = Path(parent_experiment)/'selected_task_validation.json' if parent_experiment else None
        if parent_file and parent_file.exists():
            self.parent_reference=json.loads(parent_file.read_text())
            assert self.parent_reference['trajectory_seeds']==self.seeds
            assert {(r['clip_id'],r['seed']) for r in self.parent_reference['records']}==set(self.dense)
            assert all(r['initial_noise_sha256']==self.dense[(r['clip_id'],r['seed'])]['initial_noise_sha256'] for r in self.parent_reference['records'])
        self.examples = {}
        cache_files = sorted((self.directory/'teacher_prefix_cache').glob('*.pt'))
        for path in cache_files:
            example = torch.load(path, weights_only=True, map_location='cpu')
            self.examples[example['clip_id']] = example
        assert set(self.examples)==set(self.clips)
        write_json(self.output/'task_validation_reference.json', {
            'directory':str(self.directory), 'result_sha256':file_digest(self.directory/'result.json'),
            'prefix_cache_sha256':{str(p):file_digest(p) for p in cache_files},
            'parent_selected_task_sha256':file_digest(parent_file) if self.parent_reference else None,
            'parent_selected_step':self.parent_reference['step'] if self.parent_reference else None,
            'step_zero_reproduces_parent':check_parent_reproduction,
            'clips':self.clips,'trajectory_seeds':self.seeds,'text_seed':42,
            'protocol':'Student-generated text; standardized full-prefix KV; independent paired trajectory seeds',
            'selection_uses_holdout':False, 'dense_mean_ADE_m':statistics.mean(r['ADE_m'] for r in self.rows)})

    @torch.no_grad()
    def evaluate(self, model, interface, adapters, base_weights, step):
        from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
        destination = self.output/'task_validation'/f'step_{step:06d}'
        destination.mkdir(parents=True,exist_ok=False)
        records=[]
        # Text sampling resets RNG internally: isolate it from training initialization.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), exported_weights(adapters,base_weights):
            signature=mlp_tensor_digest(model)
            for clip in self.clips:
                example=self.examples[clip]
                generated,actual=generate_prefix(model,example)
                del actual
                inputs={k:v.cuda() for k,v in generated['inputs'].items()}
                data=load_physical_aiavdataset(clip,t0_us=5_100_000,avdi=interface,maybe_stream=False)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    output=model.vlm.model(**inputs,use_cache=True,return_dict=True)
                for seed in self.seeds:
                    record={'clip_id':clip,'seed':seed,'prefix_sha256':digest(inputs['input_ids']),**sampled(model,output,inputs,data,seed)}
                    dense=self.dense[(clip,seed)]
                    assert record['initial_noise_sha256']==dense['initial_noise_sha256']
                    record['dense_ADE_m']=dense['ADE_m']
                    record['paired_ADE_difference_m']=record['ADE_m']-dense['ADE_m']
                    records.append(record)
                write_json(destination/'progress.json',{'step':step,'records':records})
                print('CONTROLLED_TASK_CLIP',step,clip,statistics.mean(r['ADE_m'] for r in records[-3:]),flush=True)
                del output,inputs,data
        mean=statistics.mean(r['ADE_m'] for r in records)
        if step == 0 and self.check_parent_reproduction:
            parent_rows=self.parent_reference['records'] if self.parent_reference else [r for r in self.reference['records'] if r['model']=='student8' and r['prefix']=='student8']
            expected=statistics.mean(r['ADE_m'] for r in parent_rows)
            if self.parent_reference and signature!=self.parent_reference['mlp_tensor_sha256']:
                raise RuntimeError('Parent checkpoint MLP SHA differs from frozen selected checkpoint')
            if abs(mean-expected)>1e-4:
                raise RuntimeError(f'Untrained recovery baseline differs from frozen parent: {mean} vs {expected}')
        result={'step':step,'mean_minADE_m':mean,'mean_ADE_m':mean,'records':records,
            'dense_mean_ADE_m':statistics.mean(r['dense_ADE_m'] for r in records),
            'missing_markers':0,'mlp_tensor_sha256':signature,'selection_uses_holdout':False,
            'trajectory_seeds':self.seeds,'evaluation_dtype':'Actual merged BF16; temporary export restored',
            'metric_semantics':'Mean ADE over 20 clips and 3 independent paired seeds; not best-of-three minADE',
            'protocol':'controlled full-prefix KV with student-generated text'}
        write_json(destination/'result.json',result)
        print('CONTROLLED_TASK_VALIDATION',step,mean,flush=True)
        return mean,result
