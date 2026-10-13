import io,tarfile,unittest
from prepare_training_chunk import FILES,safe_members,frozen_partition


class DeliveryTests(unittest.TestCase):
    def archive(self,names):
        data=io.BytesIO()
        with tarfile.open(fileobj=data,mode='w') as tar:
            for name in names:
                info=tarfile.TarInfo(name);info.size=1;tar.addfile(info,io.BytesIO(b'x'))
        data.seek(0);return tarfile.open(fileobj=data)

    def test_only_whitelisted_files(self):
        with self.archive(FILES+['download_manifest.json']) as tar:self.assertEqual(len(safe_members(tar)),12)
        for extra in ['../escape','token','features.csv']:
            with self.archive(FILES+['download_manifest.json',extra]) as tar:
                with self.assertRaises(ValueError):safe_members(tar)

    def test_partition_rejects_evaluation_overlap(self):
        clips=[f'clip{i:03d}' for i in range(100)]
        with self.assertRaises(ValueError):frozen_partition(clips,[clips[4]])
        a=frozen_partition(clips,[]);b=frozen_partition(list(reversed(clips)),[])
        self.assertEqual(a,b);self.assertEqual(len(a['training']),80);self.assertEqual(len(a['monitor_validation']),20)
        self.assertFalse(set(a['training'])&set(a['monitor_validation']))


if __name__=='__main__':unittest.main()
