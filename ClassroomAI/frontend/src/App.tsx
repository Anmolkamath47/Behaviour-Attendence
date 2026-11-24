import { useState } from 'react';
import {
  Users,
  CheckCircle,
  AlertTriangle,
  Clock,
  ShieldCheck,
  FileSpreadsheet,
  LayoutGrid,
  BarChart3,
  Info,
  EyeOff
} from 'lucide-react';

interface StudentAttendance {
  id: string;
  rollNumber: string;
  name: string;
  verifiedMinutes: number;
  totalMinutes: number;
  presencePct: number;
  status: 'PRESENT' | 'BELOW_REQUIRED_ATTENDANCE' | 'NEEDS_MANUAL_REVIEW';
  arrivalTime: string;
  departureTime: string;
  manualOverride: boolean;
  consentGiven: boolean;
  seat: string;
  primaryActivity: string;
}

const INITIAL_STUDENTS: StudentAttendance[] = [
  { id: '1', rollNumber: 'CS-01', name: 'Aarav Sharma', verifiedMinutes: 52, totalMinutes: 60, presencePct: 86.7, status: 'PRESENT', arrivalTime: '09:02 AM', departureTime: '10:00 AM', manualOverride: false, consentGiven: true, seat: 'R1-B1', primaryActivity: 'Facing Front' },
  { id: '2', rollNumber: 'CS-02', name: 'Diya Patel', verifiedMinutes: 49, totalMinutes: 60, presencePct: 81.6, status: 'PRESENT', arrivalTime: '09:05 AM', departureTime: '10:00 AM', manualOverride: false, consentGiven: true, seat: 'R1-B2', primaryActivity: 'Writing Notes' },
  { id: '3', rollNumber: 'CS-03', name: 'Ishaan Verma', verifiedMinutes: 38, totalMinutes: 60, presencePct: 63.3, status: 'BELOW_REQUIRED_ATTENDANCE', arrivalTime: '09:22 AM', departureTime: '10:00 AM', manualOverride: false, consentGiven: true, seat: 'R2-B1', primaryActivity: 'Facing Front' },
  { id: '4', rollNumber: 'CS-04', name: 'Ananya Iyer', verifiedMinutes: 46, totalMinutes: 60, presencePct: 76.7, status: 'PRESENT', arrivalTime: '09:01 AM', departureTime: '10:00 AM', manualOverride: false, consentGiven: true, seat: 'R2-B2', primaryActivity: 'Writing Notes' },
  { id: '5', rollNumber: 'CS-05', name: 'Rohan Gupta', verifiedMinutes: 28, totalMinutes: 60, presencePct: 46.7, status: 'NEEDS_MANUAL_REVIEW', arrivalTime: '09:15 AM', departureTime: '09:50 AM', manualOverride: false, consentGiven: true, seat: 'R3-B3', primaryActivity: 'Head Down Posture' },
  { id: '6', rollNumber: 'CS-06', name: 'Sneha Nair', verifiedMinutes: 55, totalMinutes: 60, presencePct: 91.7, status: 'PRESENT', arrivalTime: '09:00 AM', departureTime: '10:00 AM', manualOverride: false, consentGiven: true, seat: 'R4-B1', primaryActivity: 'Raising Hand' },
  { id: '7', rollNumber: 'CS-07', name: 'Kabir Mehta', verifiedMinutes: 44, totalMinutes: 60, presencePct: 73.3, status: 'NEEDS_MANUAL_REVIEW', arrivalTime: '09:08 AM', departureTime: '10:00 AM', manualOverride: false, consentGiven: false, seat: 'R4-B2', primaryActivity: 'Facing Front' },
  { id: '8', rollNumber: 'CS-08', name: 'Meera Rao', verifiedMinutes: 51, totalMinutes: 60, presencePct: 85.0, status: 'PRESENT', arrivalTime: '09:03 AM', departureTime: '10:00 AM', manualOverride: false, consentGiven: true, seat: 'R5-B1', primaryActivity: 'Writing Notes' },
];

export default function App() {
  const [students, setStudents] = useState<StudentAttendance[]>(INITIAL_STUDENTS);
  const [activeTab, setActiveTab] = useState<'attendance' | 'behaviour' | 'seating' | 'consent' | 'audit'>('attendance');
  const [selectedStudent, setSelectedStudent] = useState<StudentAttendance | null>(null);
  const [overrideReason, setOverrideReason] = useState('');

  // Metrics
  const totalCount = 64; // Classroom benchmark
  const presentCount = students.filter(s => s.status === 'PRESENT').length;
  const belowCount = students.filter(s => s.status === 'BELOW_REQUIRED_ATTENDANCE').length;
  const reviewCount = students.filter(s => s.status === 'NEEDS_MANUAL_REVIEW').length;
  const classAvgPct = Math.round(students.reduce((acc, s) => acc + s.presencePct, 0) / students.length);

  const handleOverride = (newStatus: 'PRESENT' | 'BELOW_REQUIRED_ATTENDANCE') => {
    if (!selectedStudent || !overrideReason) return;
    setStudents(prev => prev.map(s => {
      if (s.id === selectedStudent.id) {
        return {
          ...s,
          status: newStatus,
          manualOverride: true
        };
      }
      return s;
    }));
    setSelectedStudent(null);
    setOverrideReason('');
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      {/* Top Professional Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center text-white font-bold text-lg shadow-sm">
              C
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-semibold text-slate-900 text-base">ClassroomAI</span>
                <span className="text-xs bg-blue-50 text-blue-700 px-2 py-0.5 rounded-full font-medium border border-blue-200">
                  Room 402 • Rear Camera
                </span>
              </div>
              <p className="text-xs text-slate-500">Lecture: CS-401 Advanced Computer Architecture (60m)</p>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-1.5 text-xs text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded-md border border-emerald-200 font-medium">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              <span>LIVE OBSERVER ACTIVE</span>
            </div>
            <button
              onClick={() => alert("CSV Exporting: attendance_session_cs401.csv")}
              className="inline-flex items-center space-x-1.5 text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 px-3 py-1.5 rounded-md font-medium transition"
            >
              <FileSpreadsheet className="w-3.5 h-3.5" />
              <span>Export CSV</span>
            </button>
            <div className="text-xs text-slate-500 pl-2 border-l border-slate-200">
              Dr. R. K. Wilson
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Container */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 flex-1 w-full space-y-6">
        
        {/* Metric Cards Row */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
            <div className="flex items-center justify-between text-slate-500 mb-1">
              <span className="text-xs font-medium uppercase tracking-wider">Class Size</span>
              <Users className="w-4 h-4 text-slate-400" />
            </div>
            <div className="text-2xl font-bold text-slate-900">{totalCount}</div>
            <p className="text-xs text-slate-500 mt-1">Full rear-bench observation</p>
          </div>

          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
            <div className="flex items-center justify-between text-emerald-600 mb-1">
              <span className="text-xs font-medium uppercase tracking-wider">Present (&ge;75%)</span>
              <CheckCircle className="w-4 h-4 text-emerald-500" />
            </div>
            <div className="text-2xl font-bold text-emerald-600">{presentCount}</div>
            <p className="text-xs text-slate-500 mt-1">Verified physical presence</p>
          </div>

          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
            <div className="flex items-center justify-between text-amber-600 mb-1">
              <span className="text-xs font-medium uppercase tracking-wider">Manual Review</span>
              <AlertTriangle className="w-4 h-4 text-amber-500" />
            </div>
            <div className="text-2xl font-bold text-amber-600">{reviewCount}</div>
            <p className="text-xs text-slate-500 mt-1">Gaps or non-biometric</p>
          </div>

          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
            <div className="flex items-center justify-between text-rose-600 mb-1">
              <span className="text-xs font-medium uppercase tracking-wider">Below 75%</span>
              <Clock className="w-4 h-4 text-rose-500" />
            </div>
            <div className="text-2xl font-bold text-rose-600">{belowCount}</div>
            <p className="text-xs text-slate-500 mt-1">&lt; 45 mins in lecture</p>
          </div>

          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
            <div className="flex items-center justify-between text-blue-600 mb-1">
              <span className="text-xs font-medium uppercase tracking-wider">Verified Avg</span>
              <BarChart3 className="w-4 h-4 text-blue-500" />
            </div>
            <div className="text-2xl font-bold text-blue-600">{classAvgPct}%</div>
            <p className="text-xs text-slate-500 mt-1">Class presence score</p>
          </div>
        </div>

        {/* Ethical Separation Callout */}
        <div className="bg-blue-50/70 border border-blue-200 rounded-lg p-3 flex items-start space-x-3 text-xs text-blue-900">
          <Info className="w-4 h-4 text-blue-600 flex-shrink-0 mt-0.5" />
          <div className="leading-relaxed">
            <strong className="font-semibold text-blue-950">Ethical AI Architecture Guarantee:</strong> Physical presence and attentiveness are treated as strictly independent measurements. A student writing notes or looking down is verified present and never penalized on behavioural grounds.
          </div>
        </div>

        {/* Tabs Bar */}
        <div className="flex space-x-1 border-b border-slate-200 text-sm font-medium">
          <button
            onClick={() => setActiveTab('attendance')}
            className={`pb-3 px-4 flex items-center space-x-2 border-b-2 font-medium ${
              activeTab === 'attendance'
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            <CheckCircle className="w-4 h-4" />
            <span>Attendance (75% Rule)</span>
          </button>

          <button
            onClick={() => setActiveTab('behaviour')}
            className={`pb-3 px-4 flex items-center space-x-2 border-b-2 font-medium ${
              activeTab === 'behaviour'
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            <BarChart3 className="w-4 h-4" />
            <span>Behaviour Analytics</span>
          </button>

          <button
            onClick={() => setActiveTab('seating')}
            className={`pb-3 px-4 flex items-center space-x-2 border-b-2 font-medium ${
              activeTab === 'seating'
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            <LayoutGrid className="w-4 h-4" />
            <span>Seating Matrix (60-70)</span>
          </button>

          <button
            onClick={() => setActiveTab('consent')}
            className={`pb-3 px-4 flex items-center space-x-2 border-b-2 font-medium ${
              activeTab === 'consent'
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            <ShieldCheck className="w-4 h-4" />
            <span>Consent & Opt-Out Registry</span>
          </button>
        </div>

        {/* Tab 1: Attendance Table */}
        {activeTab === 'attendance' && (
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold text-slate-900">Verified Presence Records</h2>
                <p className="text-xs text-slate-500">Attendance decision threshold: 75% (&ge; 45 mins / 60 min session)</p>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-50 text-slate-500 uppercase tracking-wider font-medium border-b border-slate-100">
                  <tr>
                    <th className="py-3 px-4">Roll No</th>
                    <th className="py-3 px-4">Student Name</th>
                    <th className="py-3 px-4">Seat</th>
                    <th className="py-3 px-4">Presence Time</th>
                    <th className="py-3 px-4">75% Progress</th>
                    <th className="py-3 px-4">Status</th>
                    <th className="py-3 px-4">Arrival / Departure</th>
                    <th className="py-3 px-4 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {students.map((student) => (
                    <tr key={student.id} className="hover:bg-slate-50/70 transition">
                      <td className="py-3.5 px-4 font-mono font-medium text-slate-700">{student.rollNumber}</td>
                      <td className="py-3.5 px-4 font-medium text-slate-900 flex items-center space-x-2">
                        <span>{student.name}</span>
                        {!student.consentGiven && (
                          <span title="Opted out of biometrics; alternative manual seating mode" className="inline-flex text-slate-400">
                            <EyeOff className="w-3.5 h-3.5" />
                          </span>
                        )}
                        {student.manualOverride && (
                          <span className="text-[10px] bg-purple-50 text-purple-700 border border-purple-200 px-1.5 py-0.5 rounded font-normal">
                            Overridden
                          </span>
                        )}
                      </td>
                      <td className="py-3.5 px-4 text-slate-600 font-mono text-[11px]">{student.seat}</td>
                      <td className="py-3.5 px-4 text-slate-700">
                        <span className="font-semibold">{student.verifiedMinutes}m</span>
                        <span className="text-slate-400"> / {student.totalMinutes}m</span>
                      </td>
                      <td className="py-3.5 px-4 w-44">
                        <div className="flex items-center space-x-2">
                          <div className="flex-1 bg-slate-100 rounded-full h-2 overflow-hidden">
                            <div
                              className={`h-full rounded-full ${
                                student.presencePct >= 75 ? 'bg-emerald-500' : 'bg-rose-500'
                              }`}
                              style={{ width: `${Math.min(100, student.presencePct)}%` }}
                            ></div>
                          </div>
                          <span className="font-mono text-[11px] text-slate-600 w-9 text-right font-medium">
                            {student.presencePct}%
                          </span>
                        </div>
                      </td>
                      <td className="py-3.5 px-4">
                        {student.status === 'PRESENT' && (
                          <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
                            PRESENT
                          </span>
                        )}
                        {student.status === 'BELOW_REQUIRED_ATTENDANCE' && (
                          <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-rose-50 text-rose-700 border border-rose-200">
                            BELOW REQUIRED
                          </span>
                        )}
                        {student.status === 'NEEDS_MANUAL_REVIEW' && (
                          <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-amber-50 text-amber-700 border border-amber-200">
                            NEEDS REVIEW
                          </span>
                        )}
                      </td>
                      <td className="py-3.5 px-4 text-slate-500 font-mono text-[11px]">
                        {student.arrivalTime} - {student.departureTime}
                      </td>
                      <td className="py-3.5 px-4 text-right">
                        <button
                          onClick={() => setSelectedStudent(student)}
                          className="text-xs text-blue-600 hover:text-blue-800 font-medium"
                        >
                          Review / Override
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Tab 2: Behaviour Analytics */}
        {activeTab === 'behaviour' && (
          <div className="space-y-4">
            <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-slate-900">Observable Classroom Activity Distribution</h3>
                  <p className="text-xs text-slate-500">Aggregated TSM temporal shifts and XGBoost multi-modal classification across 64 students</p>
                </div>
              </div>

              {/* 8 Behaviour Classes Grid */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-2">
                {[
                  { name: 'Facing Teaching Area', pct: 48, color: 'bg-blue-500' },
                  { name: 'Writing / Notes', pct: 28, color: 'bg-emerald-500' },
                  { name: 'Raising Hand', pct: 4, color: 'bg-amber-500' },
                  { name: 'Using Mobile Phone', pct: 5, color: 'bg-rose-500' },
                  { name: 'Head-Down Posture', pct: 6, color: 'bg-indigo-500' },
                  { name: 'Looking Away', pct: 5, color: 'bg-purple-500' },
                  { name: 'Head Resting on Hand', pct: 3, color: 'bg-cyan-500' },
                  { name: 'Other / Unobservable', pct: 1, color: 'bg-slate-400' },
                ].map((act) => (
                  <div key={act.name} className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                    <div className="flex items-center justify-between text-xs text-slate-600 mb-1">
                      <span className="font-medium truncate">{act.name}</span>
                      <span className="font-bold">{act.pct}%</span>
                    </div>
                    <div className="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden">
                      <div className={`h-full ${act.color}`} style={{ width: `${act.pct}%` }}></div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Individual Student Observation Timeline */}
              <div className="pt-4 border-t border-slate-100">
                <h4 className="text-xs font-semibold text-slate-800 mb-2">Student Observation Timeline (Diya Patel - CS-02)</h4>
                <div className="flex h-7 rounded-md overflow-hidden text-[10px] text-white font-medium">
                  <div className="bg-blue-600 flex items-center justify-center" style={{ width: '45%' }} title="Facing Front: 27 mins">Facing Front (45%)</div>
                  <div className="bg-emerald-600 flex items-center justify-center" style={{ width: '40%' }} title="Writing: 24 mins">Writing (40%)</div>
                  <div className="bg-amber-500 flex items-center justify-center" style={{ width: '10%' }} title="Raised Hand: 6 mins">Hand (10%)</div>
                  <div className="bg-slate-400 flex items-center justify-center" style={{ width: '5%' }} title="Other: 3 mins">5%</div>
                </div>
                <div className="flex justify-between text-[11px] text-slate-400 mt-1 font-mono">
                  <span>00:00 (Start)</span>
                  <span>15:00</span>
                  <span>30:00 (Mid)</span>
                  <span>45:00</span>
                  <span>60:00 (End)</span>
                </div>
              </div>

              <div className="p-3 bg-amber-50/60 border border-amber-200/60 rounded-md text-[11px] text-amber-900 leading-relaxed">
                <strong>Notice:</strong> Behaviour observations are visual estimates of physical posture and hand movement. They do not represent cognitive focus, understanding, or mental attention.
              </div>
            </div>
          </div>
        )}

        {/* Tab 3: Seating Matrix */}
        {activeTab === 'seating' && (
          <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
            <div>
              <h3 className="text-sm font-semibold text-slate-900">Classroom Seating Grid (Rows 1–7, Benches A–J)</h3>
              <p className="text-xs text-slate-500">Perspective: Rear-positioned camera observing 60–70 student benches</p>
            </div>

            <div className="grid grid-cols-6 gap-2 pt-2">
              {Array.from({ length: 36 }).map((_, i) => {
                const row = Math.floor(i / 6) + 1;
                const bench = (i % 6) + 1;
                const isOccupied = i < 32;
                return (
                  <div
                    key={i}
                    className={`p-2.5 rounded-lg border text-center transition ${
                      isOccupied
                        ? 'bg-blue-50/50 border-blue-200 text-blue-900'
                        : 'bg-slate-50 border-dashed border-slate-200 text-slate-400'
                    }`}
                  >
                    <div className="text-[10px] font-mono text-slate-400">R{row}-B{bench}</div>
                    <div className="text-xs font-semibold mt-0.5">
                      {isOccupied ? `Student ${i + 1}` : 'Empty'}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Tab 4: Consent & Opt-Out Registry */}
        {activeTab === 'consent' && (
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-100">
              <h3 className="text-sm font-semibold text-slate-900">Biometric Informed Consent Registry</h3>
              <p className="text-xs text-slate-500">GDPR & Ethical Compliance: Students may opt out without penalty; non-biometric alternatives are provided.</p>
            </div>

            <div className="divide-y divide-slate-100 text-xs">
              {students.map(s => (
                <div key={s.id} className="p-4 flex items-center justify-between">
                  <div>
                    <div className="font-semibold text-slate-900">{s.name} ({s.rollNumber})</div>
                    <div className="text-slate-500 text-[11px] mt-0.5">
                      Assigned Alternative: {s.consentGiven ? 'None (Consented)' : 'Manual Seating Check & Teacher Roll Call'}
                    </div>
                  </div>
                  <div>
                    {s.consentGiven ? (
                      <span className="bg-emerald-50 text-emerald-700 border border-emerald-200 px-2.5 py-1 rounded-full text-xs font-medium">
                        Consent Granted
                      </span>
                    ) : (
                      <span className="bg-slate-100 text-slate-700 border border-slate-200 px-2.5 py-1 rounded-full text-xs font-medium">
                        Opted Out (Alternative Active)
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

      </main>

      {/* Manual Review Modal */}
      {selectedStudent && (
        <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-xl max-w-md w-full p-6 shadow-xl border border-slate-200 space-y-4">
            <div>
              <h3 className="text-base font-semibold text-slate-900">Teacher Manual Review Override</h3>
              <p className="text-xs text-slate-500 mt-1">
                Student: {selectedStudent.name} ({selectedStudent.rollNumber}) • Current: {selectedStudent.status}
              </p>
            </div>

            <div className="space-y-2 text-xs">
              <div className="p-3 bg-slate-50 rounded-lg text-slate-700 space-y-1">
                <div>Verified Presence: <strong>{selectedStudent.verifiedMinutes} mins</strong> ({selectedStudent.presencePct}%)</div>
                <div>Arrival: <strong>{selectedStudent.arrivalTime}</strong> • Departure: <strong>{selectedStudent.departureTime}</strong></div>
              </div>

              <div>
                <label className="block font-medium text-slate-700 mb-1">Audit Justification Reason</label>
                <textarea
                  value={overrideReason}
                  onChange={(e) => setOverrideReason(e.target.value)}
                  placeholder="e.g. Student sat in rear blind spot row 7; verified assignment submission."
                  className="w-full text-xs p-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
                  rows={3}
                ></textarea>
              </div>
            </div>

            <div className="flex justify-end space-x-2 pt-2">
              <button
                onClick={() => setSelectedStudent(null)}
                className="px-3.5 py-2 text-xs font-medium text-slate-600 hover:bg-slate-100 rounded-lg transition"
              >
                Cancel
              </button>
              <button
                disabled={!overrideReason.trim()}
                onClick={() => handleOverride('PRESENT')}
                className="px-3.5 py-2 text-xs font-medium bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg transition disabled:opacity-50"
              >
                Mark Present
              </button>
              <button
                disabled={!overrideReason.trim()}
                onClick={() => handleOverride('BELOW_REQUIRED_ATTENDANCE')}
                className="px-3.5 py-2 text-xs font-medium bg-rose-600 hover:bg-rose-700 text-white rounded-lg transition disabled:opacity-50"
              >
                Mark Below Required
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
