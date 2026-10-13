"""准备脚本的公开CPU边界；小夹具不能从生产CLI解除来源门禁。"""
import json
import ast
import csv
import copy
import hashlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pandas as pd
import prepare_alp15_precision_holdout as prep
from prepare_alp15_precision_holdout import resolve_chunk_path, census_usage

A = '11111111-1111-4111-8111-111111111111'
B = '22222222-2222-4222-8222-222222222222'
C = '33333333-3333-4333-8333-333333333333'
UNKNOWN = '99999999-9999-4999-8999-999999999999'
SCRIPT = Path(prep.__file__).resolve()


def fixture_frames():
    index = pd.Index([A, B, C], name='clip_id')
    clips = pd.DataFrame({'chunk': [7, 8, 8], 'clip_is_valid': [True] * 3,
                          'split': ['test', 'test', 'train']}, index=index)
    presence = pd.DataFrame({feature: [True] * 3 for feature in prep.FEATURES}, index=index)
    collection = pd.DataFrame({column: ['fixture'] * 3 for column in prep.COLLECTION}, index=index)
    return clips, presence, collection


def templates():
    return {feature: 'synthetic/' + feature + '.chunk_{chunk_id:04d}.zip' for feature in prep.FEATURES}


def blob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


class TemplateTests(unittest.TestCase):
    def test_verified_template_retains_original_directory_and_padding(self):
        self.assertEqual(resolve_chunk_path('labels/egomotion/chunk_{chunk_id:04d}.zip',
                                           7, {7}),
                         'labels/egomotion/chunk_0007.zip')

    def test_all_invalid_fields_specs_and_paths(self):
        cases = ['x.zip', 'x_{chunk_id}_{chunk_id}.zip', 'x_{other}.zip', 'x_{}.zip',
                 'x_{0}.zip', 'x_{chunk_id.real}.zip', 'x_{chunk_id[0]}.zip',
                 'x_{chunk_id!r}.zip', 'x_{chunk_id!s}.zip', 'x_{chunk_id!a}.zip',
                 'x_{chunk_id:{width}d}.zip', 'x_{chunk_id:100000d}.zip',
                 'x_{chunk_id:13d}.zip', 'x_{chunk_id:>4d}.zip', 'x_{chunk_id:4x}.zip',
                 '/x_{chunk_id}.zip', '../x_{chunk_id}.zip', './x_{chunk_id}.zip',
                 'x//x_{chunk_id}.zip', 'x/../x_{chunk_id}.zip', 'x\\x_{chunk_id}.zip',
                 'https://x/{chunk_id}.zip', 'C:/x_{chunk_id}.zip', 'x_{chunk_id}.zip?x',
                 'x_{chunk_id}.zip#x', 'x_{chunk_id}\0.zip', 'x_{chunk_id}\n.zip',
                 ' x_{chunk_id}.zip', 'x_{chunk_id}.zip ', 'x_{chunk_id}.tar',
                 '', None, float('nan'), 'x_{{chunk_id}}.zip']
        for template in cases:
            with self.subTest(template=repr(template)), self.assertRaises(ValueError):
                resolve_chunk_path(template, 7, {7})

    def test_chunks_reject_bool_float_string_negative_and_absent(self):
        for chunk in (True, 7.0, '7', -1, 8):
            with self.subTest(chunk=chunk), self.assertRaises(ValueError):
                resolve_chunk_path('x_{chunk_id:04d}.zip', chunk, {7})

    def test_csv_identity_and_structure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'features.csv'
            def save(rows):
                with path.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, ('feature', 'chunk_path'))
                    writer.writeheader()
                    writer.writerows(rows)
                return path.read_bytes()
            rows = [{'feature': f, 'chunk_path': t} for f, t in templates().items()]
            raw = save(rows)
            parsed, info = prep.read_templates(path, len(raw), blob(raw))
            self.assertEqual(parsed, templates())
            self.assertEqual(info['sha256'], hashlib.sha256(raw).hexdigest())
            for size, expected_blob in ((len(raw)+1, blob(raw)), (len(raw), '0'*40)):
                with self.assertRaises(prep.PreparationError):
                    prep.read_templates(path, size, expected_blob)
            for bad in (rows + [rows[0]], rows[:-1], [{**row, 'chunk_path': ''} for row in rows]):
                raw = save(bad)
                with self.assertRaises(prep.PreparationError):
                    prep.read_templates(path, len(raw), blob(raw))

    def test_plan_collision_and_symlink_escape_without_binary_read(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            with self.assertRaises(prep.PreparationError):
                prep.file_plan([7], {7}, {f: 'same_{chunk_id}.zip' for f in prep.FEATURES}, {'sha256': 'x'}, root)
            (root / 'synthetic').symlink_to(Path(outside), target_is_directory=True)
            with self.assertRaises(prep.PreparationError):
                prep.file_plan([7], {7}, templates(), {'sha256': 'x'}, root)

    def test_plan_receipt_requires_real_bound_path_and_keeps_quote_null(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve(); metadata=root/'files'; metadata.mkdir()
            template=templates(); target=metadata/resolve_chunk_path(template[prep.FEATURES[0]],7,{7})
            target.parent.mkdir(parents=True); target.write_bytes(b'UNREAD_ZIP')
            receipt={'local_path':str(target.relative_to(root)),'reported_bytes':10,
                     'reported_sha256':'a'*64,'source':'receipt.json'}
            plan=prep.file_plan([7],{7},template,{'sha256':'fixture'},metadata,root,[receipt])
            first=plan[0]
            self.assertEqual(first['receipt_status'],'bound_receipt_payload_sha_unverified')
            self.assertEqual(first['local_sha256'],'a'*64)
            self.assertIsNone(first['official_bytes'])
            self.assertIsNone(first['official_sha256'])
            self.assertEqual(first['quote_status'],'pending_fixed_revision_query')
            receipt['reported_bytes']=999
            self.assertEqual(prep.file_plan([7],{7},template,{'sha256':'fixture'},metadata,root,[receipt])[0]['receipt_status'],
                             'bound_receipt_missing_or_size_mismatch')


class MetadataTests(unittest.TestCase):
    def test_v3_real_parquet_radar_null_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); frames=fixture_frames(); frames[2]['radar_config']=[None,'nan','unknown']
            expected={}
            for relative,frame in zip(prep.METADATA_IDENTITIES,frames):
                path=root/relative; path.parent.mkdir(parents=True,exist_ok=True); frame.to_parquet(path)
                info=prep.identity(path); expected[relative]=(info['bytes'],info['sha256'])
            metadata,_=prep.load_metadata(root,expected)
            candidates,summary=prep.eligibility(metadata,{0,3119})
            self.assertEqual(len(candidates),3)
            self.assertTrue(pd.isna(metadata.at[A,'radar_config']))
            self.assertEqual(metadata.at[B,'radar_config'],'nan')
            self.assertEqual(metadata.at[C,'radar_config'],'unknown')
            self.assertEqual(summary['all_metadata']['missing_by_column']['radar_config'],1)
            self.assertEqual(summary['all_metadata']['by_radar_config'],{'nan':1,'unknown':1})

    def test_v3_radar_missing_preserved_and_does_not_reduce_eligibility(self):
        frames=fixture_frames()
        frames[2]['radar_config']=[None,'nan','unknown']
        original=frames[2]['radar_config'].copy()
        metadata=prep.normalize_metadata(*frames)
        pd.testing.assert_series_equal(metadata['radar_config'],original)
        candidates,summary=prep.eligibility(metadata,{0,3119})
        self.assertEqual(len(candidates),3)
        self.assertEqual(candidates['radar_config_is_missing'].tolist(),[True,False,False])
        self.assertTrue(pd.api.types.is_bool_dtype(candidates['radar_config_is_missing'].dtype))
        for layer in summary.values():
            self.assertEqual(layer['missing_by_column']['radar_config'],1)
            self.assertNotIn('None',layer['by_radar_config'])
            for column in ('split','chunk')+prep.COLLECTION:
                self.assertEqual(sum(layer['by_'+column].values())+layer['missing_by_column'][column],layer['clips'])
        self.assertEqual(summary['all_metadata']['by_radar_config'],{'nan':1,'unknown':1})
        self.assertEqual(summary['official_test_priority_pool']['by_radar_config'],{'nan':1})

    def test_v3_csv_keeps_radar_empty_and_explicit_boolean_flag(self):
        frames=fixture_frames(); frames[2]['radar_config']=[None,'nan','unknown']
        candidates,_=prep.eligibility(prep.normalize_metadata(*frames),{0,3119})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'eligible-candidates.csv'
            candidates[list(('split','chunk')+prep.COLLECTION+('radar_config_is_missing','t0_us','state'))].to_csv(path,index_label='clip_id')
            with path.open(newline='') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual([(row['radar_config'],row['radar_config_is_missing']) for row in rows],
                             [('', 'True'),('nan','False'),('unknown','False')])

    def test_v3_only_radar_null_allowed_other_fields_and_qualification_null_fail(self):
        for column in prep.COLLECTION[:-1]:
            frames=fixture_frames(); frames[2]['radar_config']=[None]*3; frames[2][column]=[None]*3
            with self.subTest(column=column),self.assertRaises(prep.PreparationError):
                prep.normalize_metadata(*frames)
        frames=fixture_frames(); frames[2]['radar_config']=[None]*3; frames[0]['clip_is_valid']=[True,None,True]
        with self.assertRaises(prep.PreparationError):
            prep.normalize_metadata(*frames)
        frames=fixture_frames(); frames[2].drop(columns=['radar_config'],inplace=True)
        with self.assertRaises(prep.PreparationError):
            prep.normalize_metadata(*frames)

    def test_v3_production_missing_count_guard_is_fixed_and_not_cli_fixture(self):
        # 只隔离计数校验函数；不替换生产CLI的字节/SHA或实际metadata读取。
        summary={'all_metadata':{'clips':306152,'missing_by_column':{'radar_config':145391}},
                 'qualified':{'clips':306152,'missing_by_column':{'radar_config':145391}},
                 'known_chunk_only_baseline':{'clips':305952,'chunks':3144,
                                             'by_split':{'train':153425,'val':90928,'test':61599}}}
        prep.require_production_counts(summary)
        for layer in ('all_metadata','qualified'):
            bad=copy.deepcopy(summary); bad[layer]['missing_by_column']['radar_config']=145390
            with self.subTest(layer=layer),self.assertRaises(prep.PreparationError):
                prep.require_production_counts(bad)
        self.assertEqual(prep.DECISIONS['ALP15-PRECISION-HOLDOUT-PREP-DECISION-V3-20261010.md'],
                         'e53284d0ebcba64787b4f6ca4f5d60bbef8e779f62f29d7d1845b1910c1731f8')

    def test_metadata_and_csv_changes_fail_final_stability_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'features.csv'; path.write_bytes(b'fixed_fixture')
            before=prep.identity(path); metadata={'fixture.parquet':{'sha256':'a','bytes':1}}
            prep.require_stable_inputs(metadata,metadata,before,prep.identity(path))
            path.write_bytes(b'other_fixture')
            with self.assertRaises(prep.PreparationError):
                prep.require_stable_inputs(metadata,metadata,before,prep.identity(path))
            with self.assertRaises(prep.PreparationError):
                prep.require_stable_inputs(metadata,{'fixture.parquet':{'sha256':'b','bytes':1}},before,before)

    def test_real_parquet_roundtrip_with_internal_fixture_identity_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = list(prep.METADATA_IDENTITIES)
            expected = {}
            for relative, frame in zip(paths, fixture_frames()):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(path)
                info = prep.identity(path)
                expected[relative] = (info['bytes'], info['sha256'])
            merged, before = prep.load_metadata(root, expected)
            self.assertEqual(list(merged.index), [A, B, C])
            self.assertEqual(len(before), 3)
            with self.assertRaises(prep.PreparationError):
                prep.load_metadata(root)  # 固定生产身份不能被小夹具通过。

    def test_invalid_dtypes_keys_nulls_and_columns(self):
        mutations = [
            lambda a,b,c: a.__setitem__('chunk', [True, False, True]),
            lambda a,b,c: a.__setitem__('chunk', [7.0, 8.0, 8.0]),
            lambda a,b,c: a.__setitem__('chunk', ['7','8','8']),
            lambda a,b,c: a.__setitem__('chunk', [-1,8,8]),
            lambda a,b,c: a.__setitem__('clip_is_valid', [1,1,1]),
            lambda a,b,c: b.__setitem__(prep.FEATURES[0], ['True']*3),
            lambda a,b,c: b.__setitem__(prep.FEATURES[0], [True,None,True]),
            lambda a,b,c: a.__setitem__('split', ['test',None,'train']),
            lambda a,b,c: c.__setitem__('country', [None]*3),
            lambda a,b,c: b.drop(columns=[prep.FEATURES[0]], inplace=True),
            lambda a,b,c: setattr(a, 'index', pd.Index([A,A,C], name='clip_id')),
            lambda a,b,c: setattr(b, 'index', pd.Index([A,B,UNKNOWN], name='clip_id')),
            lambda a,b,c: setattr(a, 'index', pd.Index([1,2,3], name='clip_id')),
            lambda a,b,c: setattr(a, 'index', pd.Index(['AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA',B,C], name='clip_id')),
        ]
        for mutation in mutations:
            frames = fixture_frames()
            mutation(*frames)
            with self.subTest(mutation=mutation), self.assertRaises(prep.PreparationError):
                prep.normalize_metadata(*frames)

    def test_eligibility_distinguishes_baseline_from_usage_and_draft(self):
        metadata = prep.normalize_metadata(*fixture_frames())
        candidates, summary = prep.eligibility(metadata, {0,7,3119})
        self.assertEqual(summary['known_chunk_only_baseline']['clips'], 3)
        self.assertEqual(summary['after_usage_exclusion']['clips'], 2)
        self.assertEqual(list(candidates.index), [B,C])
        with self.assertRaises(prep.PreparationError):
            prep.require_production_counts(summary)
        draft = prep.draft_selection(candidates, 100, 20261009)
        self.assertEqual(draft, prep.draft_selection(candidates, 100, 20261009))
        self.assertEqual(draft['clip_ids'], [B])
        self.assertTrue(draft['insufficient'])


class CensusTests(unittest.TestCase):
    def test_uppercase_uuid_keys_values_and_cache_are_canonicalized(self):
        lower='aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
        frames=fixture_frames()
        for frame in frames:
            frame.index=pd.Index([lower,B,C],name='clip_id')
        metadata=prep.normalize_metadata(*frames)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); assets=root/'assets'; assets.mkdir()
            cache=assets/'validation_00.pt'; cache.write_bytes(b'NEVER_READ')
            (assets/'mapping.json').write_text(json.dumps({lower.upper():{'clip_id':lower.upper(),'cache_file':cache.name}}))
            result=census_usage(root,[assets],metadata)
            self.assertEqual(result['used_clip_ids'],[lower])
            self.assertEqual(result['unresolved_cache_count'],0)

    def test_model_revision_is_unknown_and_only_dataset_revision_is_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); assets=root/'assets'; assets.mkdir()
            (assets/'model.json').write_text(json.dumps({'repository':'nvidia/Alpamayo-1.5','revision':'a'*40}))
            (assets/'dataset.json').write_text(json.dumps({'repository':'nvidia/PhysicalAI-Autonomous-Vehicles','revision':'main'}))
            (assets/'other.log').write_text('dataset_revision="other"\nrevision="model_sha"\n')
            result=census_usage(root,[assets],prep.normalize_metadata(*fixture_frames()))
            model_reasons=[r['reason'] for r in result['unresolved_assets'] if r['source']=='assets/model.json']
            self.assertIn('foreign_or_unknown_revision',model_reasons)
            self.assertNotIn('revision_conflict',model_reasons)
            self.assertTrue(any(r['reason']=='revision_conflict' and r['source']=='assets/dataset.json' for r in result['unresolved_assets']))

    def test_all_mapped_references_exclude_whole_chunk_and_keep_unknowns(self):
        a = '11111111-1111-4111-8111-111111111111'
        b = '22222222-2222-4222-8222-222222222222'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / 'experiments'
            assets.mkdir()
            (assets / 'split.json').write_text(json.dumps({'training': [a]}))
            (assets / 'pilot.log').write_text('clip='+b+'\n')
            (assets / 'validation_00.pt').write_bytes(b'NEVER_READ_PAYLOAD')
            metadata = pd.DataFrame({'chunk': [7, 8]}, index=pd.Index([a, b], name='clip_id'))
            ledger = census_usage(root, [assets], metadata)
            self.assertEqual(ledger['used_clip_ids'], [a, b])
            self.assertEqual(ledger['used_chunk_ids'], [0, 7, 8, 3119])
            self.assertFalse(ledger['coverage_complete'])
            self.assertEqual(ledger['unresolved_cache_count'], 1)
            self.assertEqual(ledger['file_counts']['json'], 1)
            self.assertEqual(ledger['file_counts']['pt'], 1)

    def test_json_keys_lists_broken_json_log_dedup_and_secret_redaction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / 'assets'; assets.mkdir()
            (assets / 'nested.json').write_text(json.dumps({'SECRET_TOKEN_' + A: {'holdout': [B, A]}, 'secret': 'DO_NOT_REPORT'}))
            (assets / 'broken.json').write_text('{secret=' + C)
            (assets / 'history.log').write_text('password=DO_NOT_REPORT '+UNKNOWN+' /outside/SECRET_PATH\n')
            metadata = prep.normalize_metadata(*fixture_frames())
            result = census_usage(root, [assets, assets], metadata)
            self.assertEqual(result['used_clip_ids'], [A,B,C])
            self.assertEqual(result['used_chunk_ids'], [0,7,8,3119])
            self.assertFalse(result['all_declared_text_parsed'])
            self.assertEqual(result['file_counts']['json'], 2)
            self.assertGreater(result['unresolved_external_path_count'], 0)
            text = json.dumps(result)
            for secret in ('SECRET_TOKEN_', 'DO_NOT_REPORT', 'SECRET_PATH'):
                self.assertNotIn(secret, text)
            self.assertIn(UNKNOWN, text)

    def test_explicit_cache_mapping_symlink_and_no_payload_read(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as external:
            root = Path(directory); assets = root / 'assets'; assets.mkdir()
            cache = assets / 'validation_00.pt'; cache.write_bytes(b'NEVER_READ_PAYLOAD')
            (assets / 'dataset.chunk_0009.zip').write_bytes(b'NO_ZIP_READ')
            (assets / 'receipt.json').write_text(json.dumps({'clip_id': A, 'cache_file': cache.name}))
            (assets / 'escape').symlink_to(Path(external), target_is_directory=True)
            original_open = Path.open
            def guarded(path, *args, **kwargs):
                if path.suffix in ('.pt','.zip'):
                    raise AssertionError('binary payload opened')
                return original_open(path, *args, **kwargs)
            with mock.patch.object(Path, 'open', guarded):
                result = census_usage(root, [assets], prep.normalize_metadata(*fixture_frames()))
                plan = prep.file_plan([8], {7,8}, templates(), {'sha256': 'fixture'}, root)
            self.assertEqual(result['unresolved_cache_count'], 0)
            self.assertIn(9, result['used_chunk_ids'])
            self.assertTrue(any(r['reason']=='symlink_not_followed' for r in result['unresolved_assets']))
            self.assertEqual(len(plan), 5)

    def test_new_deleted_changed_files_detected(self):
        for action in ('create','delete','change'):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); assets=root/'assets'; assets.mkdir()
                existing=assets/'log.txt'; existing.write_text(A)
                original=prep.inventory
                calls=0
                def mutated(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        if action=='create': (assets/'new.log').write_text(B)
                        elif action=='delete': existing.unlink()
                        else: existing.write_text(B)
                    return original(*args, **kwargs)
                with mock.patch.object(prep, 'inventory', mutated):
                    result=census_usage(root,[assets],prep.normalize_metadata(*fixture_frames()))
                self.assertFalse(result['scan_stable'])

    def test_read_failure_and_declared_scope_rejection(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root=Path(directory); assets=root/'assets'; assets.mkdir()
            path=assets/'bad.txt'; path.write_bytes(b'\xff')
            result=census_usage(root,[assets],prep.normalize_metadata(*fixture_frames()))
            self.assertFalse(result['all_declared_text_parsed'])
            with self.assertRaises(prep.PreparationError):
                census_usage(root,[Path(outside)],prep.normalize_metadata(*fixture_frames()))

    def test_overlap_exclusions_and_receipt_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve(); assets=root/'assets'; child=assets/'child'; child.mkdir(parents=True)
            hidden=assets/'output'; hidden.mkdir(); (hidden/'never.json').write_text(json.dumps({'clip_id':C}))
            archive=child/'chunk_0007.zip'; archive.write_bytes(b'zip')
            receipt={'repository':'nvidia/PhysicalAI-Autonomous-Vehicles','revision':prep.REVISION,
                     'files':[{'path':'chunk_0007.zip','bytes':3,'sha256':'a'*64}]}
            (child/'receipt.json').write_text(json.dumps(receipt))
            (assets/'split.json').write_text(json.dumps({'training':[A]}))
            result=census_usage(root,[assets,child],prep.normalize_metadata(*fixture_frames()),[hidden])
            self.assertEqual(result['file_counts']['json'],2)
            self.assertEqual(result['used_clip_ids'],[A])
            self.assertEqual(result['existing_receipt_evidence'][0]['local_path'],'assets/child/chunk_0007.zip')
            self.assertTrue(any(row['kind']=='excluded' for row in result['scan_manifest']))

    def test_permission_failure_is_recorded_without_message_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); assets=root/'assets'; assets.mkdir(); blocked=assets/'private.log'; blocked.write_text(A)
            original=Path.open
            def guarded(path,*args,**kwargs):
                if path==blocked.resolve(): raise PermissionError('SENSITIVE_EXCEPTION')
                return original(path,*args,**kwargs)
            with mock.patch.object(Path,'open',guarded):
                result=census_usage(root,[assets],prep.normalize_metadata(*fixture_frames()))
            self.assertFalse(result['all_declared_text_parsed'])
            self.assertNotIn('SENSITIVE_EXCEPTION',json.dumps(result))

    def test_same_size_mtime_text_rewrite_is_detected_by_final_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); assets=root/'assets'; assets.mkdir(); existing=assets/'log.txt'; existing.write_text(A)
            original=prep.inventory; calls=0; old_stat=existing.stat()
            def mutated(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==2:
                    existing.write_text(B)
                    import os
                    os.utime(existing,ns=(old_stat.st_atime_ns,old_stat.st_mtime_ns))
                return original(*args,**kwargs)
            with mock.patch.object(prep,'inventory',mutated):
                result=census_usage(root,[assets],prep.normalize_metadata(*fixture_frames()))
            self.assertFalse(result['scan_stable'])


class CliTests(unittest.TestCase):
    def test_real_cli_bad_metadata_leaves_failure_manifest_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); metadata=root/'metadata'; metadata.mkdir()
            assets=root/'assets'; assets.mkdir()
            for relative, frame in zip(prep.METADATA_IDENTITIES, fixture_frames()):
                path=metadata/relative; path.parent.mkdir(parents=True, exist_ok=True); frame.to_parquet(path)
            output=root/'new_output'
            args=[sys.executable,'-B',str(SCRIPT),'--root',str(root),'--metadata-root',str(metadata),
                  '--usage-root',str(assets),'--output',str(output)]
            result=subprocess.run(args,text=True,capture_output=True)
            self.assertEqual(result.returncode,1,result.stderr)
            manifest=json.loads((output/'run-manifest.json').read_text())
            self.assertEqual(manifest['status'],'failed')
            self.assertEqual(manifest['environment']['physical_ai_av'],'0.2.0')
            before=prep.sha256(output/'run-manifest.json')
            self.assertEqual(subprocess.run(args,capture_output=True).returncode,2)
            self.assertEqual(before,prep.sha256(output/'run-manifest.json'))
            self.assertFalse((output/'draft-feasibility-selection.json').exists())

    def test_cli_has_no_fixture_or_coverage_override_and_no_private_import(self):
        result=subprocess.run([sys.executable,'-B',str(SCRIPT),'--help'],text=True,capture_output=True)
        self.assertEqual(result.returncode,0)
        self.assertNotIn('--fixture',result.stdout)
        self.assertNotIn('--coverage',result.stdout)
        tree=ast.parse(SCRIPT.read_text())
        imports=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): imports.extend(alias.name for alias in node.names)
            elif isinstance(node,ast.ImportFrom): imports.append(node.module or '')
        for forbidden in ('torch','pickle','physical_ai_av','huggingface_hub','requests','urllib'):
            self.assertFalse(any(name==forbidden or name.startswith(forbidden+'.') for name in imports))

    def test_output_outside_declared_root_is_not_created(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root=Path(directory); output=Path(outside)/'forbidden'
            result=subprocess.run([sys.executable,'-B',str(SCRIPT),'--root',str(root),'--metadata-root',str(root),
                                   '--usage-root',str(root),'--output',str(output)],capture_output=True)
            self.assertEqual(result.returncode,2)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
