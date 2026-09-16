export default function MediGuardMark() {
  return (
    <svg
      aria-hidden="true"
      focusable="false"
      viewBox="0 0 48 48"
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <linearGradient id="mediguard-mark-bg" x1="8%" x2="92%" y1="5%" y2="95%">
          <stop offset="0%" stopColor="#4f9be6" />
          <stop offset="48%" stopColor="#3f8fe6" />
          <stop offset="100%" stopColor="#2a6db5" />
        </linearGradient>
        <linearGradient id="mediguard-mark-edge" x1="12%" x2="88%" y1="8%" y2="92%">
          <stop offset="0%" stopColor="#ffffff" stopOpacity="0.96" />
          <stop offset="100%" stopColor="#ffffff" stopOpacity="0.72" />
        </linearGradient>
        <radialGradient id="mediguard-mark-gloss" cx="20%" cy="13%" r="82%">
          <stop offset="0%" stopColor="#86bff6" stopOpacity="0.12" />
          <stop offset="52%" stopColor="#6db0ec" stopOpacity="0.03" />
          <stop offset="100%" stopColor="#6db0ec" stopOpacity="0" />
        </radialGradient>
        <filter id="mediguard-mark-shadow" x="-20%" y="-20%" width="140%" height="150%">
          <feDropShadow
            dx="0"
            dy="1.15"
            stdDeviation="1.05"
            floodColor="#164f85"
            floodOpacity="0.24"
          />
        </filter>
        <filter id="mediguard-symbol-depth" x="-12%" y="-12%" width="124%" height="132%">
          <feDropShadow
            dx="0"
            dy="1.1"
            stdDeviation="0.75"
            floodColor="#15558f"
            floodOpacity="0.32"
          />
        </filter>
      </defs>
      <rect
        fill="url(#mediguard-mark-bg)"
        height="41"
        rx="10"
        stroke="url(#mediguard-mark-edge)"
        strokeWidth="1.05"
        width="41"
        x="3.5"
        y="3.5"
        filter="url(#mediguard-mark-shadow)"
      />
      <rect
        fill="url(#mediguard-mark-gloss)"
        height="38.8"
        rx="9.2"
        width="38.8"
        x="4.6"
        y="4.6"
      />
      <svg
        aria-hidden="true"
        height="31"
        viewBox="0 0 1149 1024"
        width="35"
        x="6.5"
        y="8.5"
      >
        <g fill="currentColor" filter="url(#mediguard-symbol-depth)">
          <path d="M729.733 226.526c-282.549-43.693-426.006 100.496-426.737 100.496C6.614 584.81 347.42 776.328 347.42 777.055c242.498 133.99 514.122 29.132 512.666 27.678 188.607-60.448 238.855-218.465 238.855-218.465s21.116 2.913 22.94 0.723c-9.829 46.608-35.889 90.404-31.68 83.743-69.908 112.875-225.746 251.234-564.365 243.23-338.62-8.01-453.677-249.048-453.677-249.048s-122.343-135.447 38.594-323.328C271.69 153.712 543.314 148.613 549.138 151.524c5.824 2.913 184.239-18.203 361.921 72.095 177.69 90.298 210.821 251.964 210.821 251.964h-27.308c-97.577-229.387-367.75-249.057-364.84-249.057z" />
          <path d="M828.047 751.685h110.32l68.82-438.014H904.51m-364.84 14.083c-127.254 44.242-119.142 111.982-118.513 114.694 16.102 69.32 69.353 65.224 126.705 141.457 78.153 103.877-23.483 122.337-24.028 122.337-84.655 10.383-131.628-5.461-131.628-5.461l-40.962-8.734v67.177s40.419 6.555 44.242 6.555 176.409 18.57 253.965-8.738 95.578-87.932 97.215-127.802c1.637-39.871-36.829-81.607-49.698-89.026-29.247-16.86-74.825-48.61-74.825-48.61s-58.44-44.786-21.847-90.116c36.595-45.33 196.618-46.424 196.618-46.424l-3.28-45.874c-132.166-18.03-255.601 20.75-253.964 18.565z" />
        </g>
      </svg>
    </svg>
  );
}
