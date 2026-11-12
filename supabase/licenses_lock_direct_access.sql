-- 프로그램에 들어 있는 공개 키로는 licenses 표를 직접 읽거나 쓰지 못하게 한다.
-- 인증 확인(verify_license)과 관리자 기능(admin_*)은 서버 함수로만 한다.
--
-- 되돌리려면:
--   alter table public.licenses disable row level security;
--   grant all on public.licenses to anon, authenticated;
--   grant execute on function public.reset_device(text) to anon, authenticated;

alter table public.licenses enable row level security;      -- 이미 있는 'Deny direct access' 정책이 이제 적용된다
revoke all on table public.licenses from anon, authenticated;

-- 누구나 아무 인증번호의 PC 묶임을 풀 수 있던 함수: 관리자용 admin_reset_device로 대신한다
revoke execute on function public.reset_device(text) from public, anon, authenticated;

alter function public.verify_license(text, text) set search_path = public;
alter function public.reset_device(text) set search_path = public;
