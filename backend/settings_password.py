"""Initialize/reset the settings password locally, never through the public website."""
import argparse
import getpass
import warnings
from pathlib import Path

from config import Settings
from settings_security import SettingsPassword
from store import Store


def main():
    parser = argparse.ArgumentParser(description='配置后台设置管理密码（不会回显或保存明文）')
    parser.add_argument('action', choices=('set', 'unlock'), help='set 初始化/重置密码；unlock 清除临时锁定')
    parser.add_argument('--data-dir', type=Path, help='数据库目录；默认 TRAFFIC_DATA_DIR 或当前目录下 data')
    args = parser.parse_args()
    settings = Settings(**({'data': args.data_dir.resolve()} if args.data_dir else {}))
    password = SettingsPassword(Store(settings))
    if args.action == 'unlock':
        password.unlock()
        print('已清除管理密码临时锁定。')
        return
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', getpass.GetPassWarning)
            first = getpass.getpass('新管理密码（15–128 个字符）: ')
            second = getpass.getpass('再次输入: ')
        if first != second:
            parser.error('两次密码不一致，未作修改。')
        password.set_password(first)
    except (ValueError, EOFError, getpass.GetPassWarning) as error:
        parser.exit(1, '无法设置密码：请在交互终端输入 15–128 个字符，两次保持一致。\n')
    print('管理密码已保存，即刻生效；网页每次保存设置都需验证。')


if __name__ == '__main__':
    main()
