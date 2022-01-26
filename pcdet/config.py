from pathlib import Path

import yaml
from easydict import EasyDict


def log_config_to_file(cfg, pre='cfg', logger=None):
	for key, val in cfg.items():
		if isinstance(cfg[key], EasyDict):
			logger.info('\n%s.%s = edict()' % (pre, key))
			log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
			continue
		logger.info('%s.%s: %s' % (pre, key, val))


def cfg_from_list(cfg_list, config):
	"""Set config keys via list (e.g., from command line)."""
	from ast import literal_eval
	assert len(cfg_list) % 2 == 0
	for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
		key_list = k.split('.')
		d = config
		for subkey in key_list[:-1]:
			assert subkey in d, 'NotFoundKey: %s' % subkey
			d = d[subkey]
		subkey = key_list[-1]
		assert subkey in d, 'NotFoundKey: %s' % subkey
		try:
			value = literal_eval(v)
		except:
			value = v
		
		if type(value) != type(d[subkey]) and isinstance(d[subkey], EasyDict):
			key_val_list = value.split(',')
			for src in key_val_list:
				cur_key, cur_val = src.split(':')
				val_type = type(d[subkey][cur_key])
				cur_val = val_type(cur_val)
				d[subkey][cur_key] = cur_val
		elif type(value) != type(d[subkey]) and isinstance(d[subkey], list):
			val_list = value.split(',')
			for k, x in enumerate(val_list):
				val_list[k] = type(d[subkey][0])(x)
			d[subkey] = val_list
		else:
			assert type(value) == type(d[subkey]), \
				'type {} does not match original type {}'.format(type(value), type(d[subkey]))
			d[subkey] = value


def merge_new_config(config, new_config):
	if '_BASE_CONFIG_' in new_config:
		# 如果新配置中有‘_BASE_CONFIG_’，则打开对应文件
		with open(new_config['_BASE_CONFIG_'], 'r') as f:
			try:
				yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
			except:
				yaml_config = yaml.safe_load(f)
		# 更新新配置到旧配置中
		config.update(EasyDict(yaml_config))
	# 如果有和_BASE_CONFIG_内相同的变量将覆盖。
	for key, val in new_config.items():
		if not isinstance(val, dict):
			# 新配置中的子项如果是值，则增加key-val对
			config[key] = val
			continue
		# 新配置的子项是字典，则为原配置创建子字典。
		if key not in config:
			config[key] = EasyDict()
		# 递归，用子项字典填充子字典
		merge_new_config(config[key], val)
	
	return config


def cfg_from_yaml_file(cfg_file, config):
	with open(cfg_file, 'r') as f:
		try:
			new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
		except:
			new_config = yaml.safe_load(f)
		
		merge_new_config(config=config, new_config=new_config)
	
	return config


cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0
