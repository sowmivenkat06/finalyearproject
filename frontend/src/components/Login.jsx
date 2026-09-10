import React, { useState } from 'react';
import { Mail, Lock, Eye, EyeOff, Sparkles, Target, ShieldCheck } from 'lucide-react';

export default function Login({ onLoginSuccess }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!email || !password) {
      setError('Please fill in all fields');
      return;
    }
    
    // Dynamically derive name and initials from the email
    const usernamePart = email.split('@')[0]; // e.g. "sowmiya" or "john.doe"
    const nameParts = usernamePart.split(/[\._\-]/).map(part => part.charAt(0).toUpperCase() + part.slice(1));
    let derivedName = nameParts.join(' ');
    
    // Default fallback to Dr. Sowmiya if sowmiya is entered
    if (usernamePart.toLowerCase() === 'sowmiya') {
      derivedName = 'Dr. Sowmiya';
    } else if (nameParts.length === 1) {
      derivedName = `Dr. ${nameParts[0]}`;
    }
    
    // Determine initials
    let initials = 'DS';
    const cleanNameForInitials = derivedName.startsWith('Dr. ') ? derivedName.substring(4) : derivedName;
    const initialsParts = cleanNameForInitials.split(' ');
    if (initialsParts.length >= 2) {
      initials = (initialsParts[0].charAt(0) + initialsParts[1].charAt(0)).toUpperCase();
    } else if (initialsParts.length === 1 && initialsParts[0].length > 0) {
      initials = initialsParts[0].substring(0, 2).toUpperCase();
    }

    onLoginSuccess({ email, name: derivedName, initials });
  };

  return (
    <div className="login-container">
      <div className="login-card-wrapper animate-fade">
        {/* Left Side: Brand Info and Assets */}
        <div className="login-left">
          <div className="login-left-content">
            <h1>
              A Fusion-Based <br />
              Deep Learning Framework <br />
              for Enhanced <span className="highlight-purple">Cervical Cancer</span> <br />
              Detection and <span className="highlight-purple">Classification</span>
            </h1>
            <p className="login-left-subtitle">Early Detection Saves Lives</p>
            
            <div className="badges-row">
              <div className="badge-item">
                <div className="badge-icon">
                  <Sparkles size={20} />
                </div>
                <span>AI Powered</span>
              </div>
              <div className="badge-item">
                <div className="badge-icon">
                  <Target size={20} />
                </div>
                <span>Accurate</span>
              </div>
              <div className="badge-item">
                <div className="badge-icon">
                  <ShieldCheck size={20} />
                </div>
                <span>Reliable</span>
              </div>
            </div>
          </div>

          <div className="illustrations-box">
            <img src="/microscope.png" alt="Microscope" className="microscope-img" />
            <div className="cells-circle-preview">
              <img src="/cells.png" alt="Cervical Cells Preview" />
            </div>
          </div>
        </div>

        {/* Right Side: Login Card */}
        <div className="login-right">
          <div className="login-right-header">
            <div className="ribbon-icon-wrapper">
              <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2c-2.8 0-5 2.2-5 5c0 2.5 1.5 4.5 4 6.5l-4 8.5c-.3.6 0 1.3.6 1.6c.6.3 1.3 0 1.6-.6l3.8-8c.5.2 1 .2 1.6.2s1.1 0 1.6-.2l3.8 8c.3.6 1 .9 1.6.6c.6-.3.9-1 .6-1.6l-4-8.5c2.5-2 4-4 4-6.5c0-2.8-2.2-5-5-5Z" />
              </svg>
            </div>
            <h2>Welcome Back</h2>
            <p>Login to access the Cervical Cancer Detection System</p>
          </div>

          <form onSubmit={handleSubmit}>
            {error && (
              <div style={{ color: 'var(--color-danger)', fontSize: '0.9rem', marginBottom: '1rem', textAlign: 'center' }}>
                {error}
              </div>
            )}
            
            <div className="form-group">
              <div className="input-icon-wrapper">
                <Mail size={18} />
              </div>
              <input
                type="email"
                placeholder="Email address"
                className="input-field"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>

            <div className="form-group">
              <div className="input-icon-wrapper">
                <Lock size={18} />
              </div>
              <input
                type={showPassword ? "text" : "password"}
                placeholder="Password"
                className="input-field"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
              <button
                type="button"
                className="eye-toggle-btn"
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>

            <button type="submit" className="login-btn">
              Login
            </button>
          </form>

          <div className="login-footer">
            Don't have an account? <a href="#contact" onClick={() => alert('Please contact administration at admin@kongu.edu for credentials.')}>Contact Admin</a>
          </div>
        </div>
      </div>
    </div>
  );
}
