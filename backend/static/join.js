'use strict';
const token = new URLSearchParams(location.search).get('token') || '';
const form = document.getElementById('joinForm');
const error = document.getElementById('joinError');
document.getElementById('server').textContent = location.origin;
if (token.length < 16) {
  error.textContent = '这个邀请链接不完整。请向管理员重新要一条。';
  form.querySelector('button').disabled = true;
}
form.onsubmit = async event => {
  event.preventDefault();
  error.textContent = '';
  const response = await fetch('/v1/register', {
    method: 'POST',
    credentials: 'same-origin',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      token,
      username: document.getElementById('username').value.trim(),
      password: document.getElementById('password').value,
    }),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    error.textContent = typeof data.detail === 'string' ? data.detail : '注册失败';
    return;
  }
  location.href = '/';
};
