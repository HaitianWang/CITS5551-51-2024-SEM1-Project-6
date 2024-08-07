<template>
  <div>
    <h1>Time Capsule</h1>
    <p>Welcome, {{ $route.query.username }}</p>
  </div>
</template>

<script>
  export default {
    name: 'TimeCapsule'
  };
</script>
