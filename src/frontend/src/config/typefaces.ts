export type TypefaceKey = 'misans' | 'noto' | 'system';

export interface TypefaceOption {
  key: TypefaceKey;
  label: string;
  note: string;
  stack: string;
  sampleWeight: number;
}

export const TYPEFACE_STORAGE_KEY = 'mediguard.typeface';

export const TYPEFACE_OPTIONS: TypefaceOption[] = [
  {
    key: 'noto',
    label: 'Noto Sans SC',
    note: '当前默认，更中性，政务感更稳',
    stack:
      '"Noto Sans SC", "MiSans", "PingFang SC", "Microsoft YaHei", sans-serif',
    sampleWeight: 600,
  },
  {
    key: 'misans',
    label: 'MiSans',
    note: '更现代，可作为备选',
    stack:
      '"MiSans", "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif',
    sampleWeight: 600,
  },
  {
    key: 'system',
    label: 'Microsoft YaHei',
    note: '最熟悉，保守稳妥，偏传统 Windows 风格',
    stack:
      '"Microsoft YaHei", "Microsoft YaHei UI", "PingFang SC", sans-serif',
    sampleWeight: 600,
  },
];

export const DEFAULT_TYPEFACE: TypefaceKey = 'noto';

export function getTypefaceOption(key: TypefaceKey) {
  return TYPEFACE_OPTIONS.find((option) => option.key === key) ?? TYPEFACE_OPTIONS[0];
}
