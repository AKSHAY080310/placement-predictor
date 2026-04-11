import { useEffect, useState } from "react";
import "../styles/placementform.css";


const PlacementForm = () => {
  console.log("PlacementForm mounted"); 
  const [formData, setFormData] = useState({
    age: "",
    gender: "",
    stream: "",
    internships: "",
    cgpa: "",
    history_of_backlogs: "",
    role: "",
    skills: []
  });

  const [genderOptions, setGenderOptions] = useState([]);
  const [streamOptions, setStreamOptions] = useState([]);
  const [backlogOptions, setBacklogOptions] = useState([]);
  const [roleOptions, setRoleOptions] = useState([]);
  const [skillOptions, setSkillOptions] = useState([]);
  const [result, setResult] = useState(null);


useEffect(() => {
  console.log("🔵 useEffect triggered");

  fetch("http://localhost:8000/options/placement")
    .then(res => {
      console.log("placement response status:", res.status);
      return res.json();
    })
    .then(data => {
      console.log("placement data:", data);
      setGenderOptions(data.gender || []);
      setStreamOptions(data.stream || []);
      setBacklogOptions(data.history_of_backlogs || []);
    })
    .catch(err => console.error("placement fetch error:", err));

  fetch("http://localhost:8000/options/roles")
    .then(res => {
      console.log("roles response status:", res.status);
      return res.json();
    })
    .then(data => {
      console.log("roles data:", data);
      setRoleOptions(data.roles || []);
    })
    .catch(err => console.error("roles fetch error:", err));
}, []);

 
  useEffect(() => {
    if (!formData.role) {
      setSkillOptions([]);
      return;
    }

    fetch(`http://localhost:8000/options/skills?role=${formData.role}`)
      .then(res => res.json())
      .then(data => setSkillOptions(data.skills || []));
  }, [formData.role]);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const toggleSkill = (skill) => {
    setFormData(prev => ({
      ...prev,
      skills: prev.skills.includes(skill)
        ? prev.skills.filter(s => s !== skill)
        : [...prev.skills, skill]
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const payload = {
      ...formData,
      age: Number(formData.age),
      internships: Number(formData.internships),
      cgpa: Number(formData.cgpa)
    };

    const response = await fetch("http://localhost:8000/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await response.json();
    setResult(data);
  };

  return (
    <form className="placement-form" onSubmit={handleSubmit}>
      <h2>Placement Predictor</h2>

      <div className="form-group">
        <label>Age</label>
        <input type="number" name="age" onChange={handleChange} />
      </div>

      <div className="form-group">
        <label>Gender</label>
        <select name="gender" onChange={handleChange}>
          <option value="">Select Gender</option>
          {genderOptions.map(g => (
            <option key={g} value={g}>{g}</option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label>Stream</label>
        <select name="stream" onChange={handleChange}>
          <option value="">Select Stream</option>
          {streamOptions.map(s => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label>Internships</label>
        <input type="number" name="internships" onChange={handleChange} />
      </div>

      <div className="form-group">
        <label>CGPA</label>
        <input type="number" step="0.01" name="cgpa" onChange={handleChange} />
      </div>

      <div className="form-group">
        <label>History of Backlogs</label>
        <select name="history_of_backlogs" onChange={handleChange}>
          <option value="">Select</option>
          {backlogOptions.map(b => (
            <option key={b} value={b}>{b}</option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label>Role</label>
        <select name="role" onChange={handleChange}>
          <option value="">Select Role</option>
          {roleOptions.map(r => (
            <option key={r} value={r}>{r}</option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label>Skills</label>
        <div className="skills-box">
          {skillOptions.map(skill => (
            <label key={skill}>
              <input
                type="checkbox"
                checked={formData.skills.includes(skill)}
                onChange={() => toggleSkill(skill)}
              />
              {skill}
            </label>
          ))}
        </div>
      </div>

      <button type="submit">Evaluate</button>

      {result && (
        <div className="result-box">
          <h3>Final Score: {result.final_score}</h3>
          <p>Placement Probability: {result.placement_probability}</p>
          <p>Skill Match Score: {result.skill_match_score}</p>

          <h4>Missing Skills</h4>
          <ul>
            {result.missing_skills.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </div>
      )}
    </form>
  );
};

export default PlacementForm;
