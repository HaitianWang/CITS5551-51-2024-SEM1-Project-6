<template>
  <div>

  </div>
</template>

<script>

    export default {
        name: "mainPage",
        created() {
          this.$emit('header', true);
          this.$emit('content', true);
          this.$emit('footer', true);
        },
    }

</script>

<style scoped>

</style>
