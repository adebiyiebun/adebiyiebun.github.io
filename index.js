const data = [1, 0.5, 0.2, 0.8, 0.75]; // Example array of values

// Create a radar chart
const ctx = document.getElementById('radarChart').getContext('2d');
new Chart(ctx, {
  type: 'radar',
  data: {
    labels: ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism'], // Replace with your own labels
    datasets: [{
      label: 'Data',
      data: data,
      backgroundColor: 'rgba(54, 162, 235, 0.5)', // Adjust the color as desired
      borderColor: 'rgba(54, 162, 235, 1)', // Adjust the color as desired
      borderWidth: 2
    }]
  },
  options: {
    scale: {
      ticks: {
        beginAtZero: true,
        max: 100,
        stepSize: 20
      }
    }
  }
});
