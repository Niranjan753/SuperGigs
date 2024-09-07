import * as tf from '@tensorflow/tfjs';

// Mock data for training
const trainingData = [
  { input: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], output: [1, 0, 0] }, // User with JavaScript skill
  { input: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], output: [1, 0, 0] }, // User with React skill
  { input: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], output: [1, 0, 0] }, // User with Node.js skill
  { input: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], output: [0, 1, 0] }, // User with Python skill
  { input: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], output: [0, 1, 0] }, // User with Java skill
  { input: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], output: [0, 1, 0] }, // User with C++ skill
  { input: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], output: [0, 0, 1] }, // User with Illustrator skill
  { input: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], output: [0, 0, 1] }, // User with Photoshop skill
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], output: [0, 0, 1] }, // User with Copywriting skill
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], output: [0, 0, 1] }, // User with SEO skill
];

// Convert data to tensors
const inputs = tf.tensor2d(trainingData.map(d => d.input));
const outputs = tf.tensor2d(trainingData.map(d => d.output));

// Create and train the model
const createModel = async () => {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [10] }));
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

  model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

  await model.fit(inputs, outputs, { epochs: 100 });

  return model;
};

export const loadModel = async () => {
  const model = await createModel();
  return model;
};

export const getRecommendations = (model, userInput) => {
  const inputTensor = tf.tensor2d([userInput], [1, userInput.length]);
  const prediction = model.predict(inputTensor);
  return prediction.dataSync();
};

// Define available gigs
const availableGigs = [
  { id: 1, title: "Web Development Project", category: "Web Development" },
  { id: 2, title: "Mobile App Design", category: "Design" },
  { id: 3, title: "Content Writing for Blog", category: "Writing" },
  { id: 4, title: "Python Data Analysis", category: "Data Science" },
  { id: 5, title: "Java Backend Development", category: "Backend Development" },
  { id: 6, title: "C++ Game Development", category: "Game Development" },
  { id: 7, title: "Logo Design", category: "Graphic Design" },
  { id: 8, title: "Photo Editing", category: "Photo Editing" },
  { id: 9, title: "Technical Writing", category: "Writing" },
  { id: 10, title: "SEO Optimization", category: "Digital Marketing" },
];

// Map prediction categories to gig categories
const categoryMapping = {
  0: ["Web Development", "Backend Development"],
  1: ["Data Science", "Game Development"],
  2: ["Design", "Graphic Design", "Photo Editing", "Writing", "Digital Marketing"],
};

export const recommendGigs = (model, userSkills) => {
  const prediction = getRecommendations(model, userSkills);
  const predictedCategory = prediction.indexOf(Math.max(...prediction));
  
  const recommendedCategories = categoryMapping[predictedCategory];
  
  const recommendedGigs = availableGigs.filter(gig => 
    recommendedCategories.includes(gig.category)
  );
  
  return recommendedGigs;
};

// Example usage:
// const model = await loadModel();
// const userSkills = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]; // Example: User skilled in JavaScript, React, and Node.js
// const recommendations = recommendGigs(model, userSkills);
// console.log(recommendations);

